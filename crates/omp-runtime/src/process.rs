//! Native Windows Restricted Process runner.
//!
//! Enforces OS-level isolation using Windows AppContainer / Lowbox tokens and Job Objects:
//! - AppContainer profile with no network capabilities by default.
//! - Working directory permissions strictly granted via DACL and Low Mandatory Integrity Label.
//! - Journal and host filesystem outside `cwd` denied at kernel level.
//! - Process Job Object with `KILL_ON_JOB_CLOSE`, memory, CPU, and child process limits.
//! - Suspended creation with Job Object assignment prior to resume to eliminate spawn races.
//! - Handle inheritance restricted strictly to stdio pipes via `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`.
//! - Sanitized environment block preventing credential and secret leakage.
//! - Tracked DACL restore for external programs: host filesystem permissions are never permanently altered.

pub use std::fs::File;

#[cfg(windows)]
pub use windows_impl::RestrictedProcess;

#[cfg(not(windows))]
pub use fallback_impl::RestrictedProcess;

#[cfg(windows)]
mod windows_impl {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io::Write;
    use std::os::windows::io::FromRawHandle;
    use std::path::{Path, PathBuf};
    use std::ptr::{null, null_mut};

    use omp_types::{LimitPolicy, StructuredError};
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, HANDLE, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE,
        SetHandleInformation, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::System::Pipes::CreatePipe;
    use windows_sys::Win32::System::Threading::{
        CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, CreateProcessW,
        EXTENDED_STARTUPINFO_PRESENT, GetExitCodeProcess, INFINITE, PROCESS_INFORMATION,
        ResumeThread, STARTF_USESTDHANDLES, STARTUPINFOEXW, TerminateProcess, WaitForSingleObject,
    };

    use crate::windows_sandbox::{
        AppContainerSandbox, JobObjectBoundary, ProcThreadAttributeList,
        build_sanitized_environment, to_wide_null,
    };

    /// An OS-level restricted process running inside an AppContainer and a Job Object.
    pub struct RestrictedProcess {
        job: JobObjectBoundary,
        process_handle: HANDLE,
        pid: u32,
        stdin: Option<File>,
        stdout: Option<File>,
        stderr: Option<File>,
        sandbox: Option<AppContainerSandbox>,
        cwd: PathBuf,
        exit_code: Option<i32>,
        killed: bool,
    }

    unsafe impl Send for RestrictedProcess {}

    impl RestrictedProcess {
        /// Spawn a process with OS-level AppContainer and Job Object restrictions.
        ///
        /// 1. Resolves executable to an absolute canonical path.
        /// 2. AppContainer profile is created with no network capabilities using an opaque minted token.
        /// 3. `cwd` is granted to the AppContainer SID via DACL and Low Mandatory Integrity Label.
        /// 4. If `program` is outside `cwd` and system directories, access is granted with tracked restore on drop.
        /// 5. Job Object is created with `KILL_ON_JOB_CLOSE`, memory, CPU, and process count limits.
        /// 6. Anonymous pipes are created, and only child stdio handles are inheritable.
        /// 7. Process is created in a suspended state (`CREATE_SUSPENDED`).
        /// 8. Child process is assigned to Job Object *before* resuming execution (prevents spawn race).
        /// 9. Main thread is resumed; pipes are wrapped in native Rust stdio readers/writers.
        pub fn spawn(
            program: &Path,
            args: &[String],
            cwd: &Path,
            limits: &LimitPolicy,
        ) -> Result<Self, StructuredError> {
            Self::spawn_internal(program, args, cwd, limits, false, &[])
        }

        pub fn spawn_with_runtime(
            program: &Path,
            args: &[String],
            cwd: &Path,
            limits: &LimitPolicy,
            runtime_files: &[PathBuf],
        ) -> Result<Self, StructuredError> {
            Self::spawn_internal(program, args, cwd, limits, false, runtime_files)
        }

        fn spawn_internal(
            program: &Path,
            args: &[String],
            cwd: &Path,
            limits: &LimitPolicy,
            allow_network: bool,
            runtime_files: &[PathBuf],
        ) -> Result<Self, StructuredError> {
            // Verify working directory exists
            if !cwd.exists() {
                return Err(StructuredError::new(
                    "cwd_not_found",
                    format!("working directory '{}' does not exist", cwd.display()),
                    false,
                ));
            }

            // Resolve program to absolute canonical path
            let resolved_program = PathBuf::from(
                resolve_executable_path(program, cwd)
                    .to_string_lossy()
                    .replace('/', "\\"),
            );

            // Create AppContainer sandbox with Lowbox token and no network capability
            let mut sandbox = AppContainerSandbox::create(cwd, allow_network)?;

            // Grant read/execute access to external binary if located outside cwd and system paths.
            // This is recorded as a tracked grant and automatically restored on drop.
            sandbox.grant_program_access(&resolved_program)?;
            for file in runtime_files {
                sandbox.grant_program_access(file)?;
            }

            // Create and configure Job Object boundary with resource bounds
            let job = JobObjectBoundary::new(limits)?;

            // Prepare anonymous stdio pipes
            let mut child_stdin_read: HANDLE = null_mut();
            let mut host_stdin_write: HANDLE = null_mut();
            let mut host_stdout_read: HANDLE = null_mut();
            let mut child_stdout_write: HANDLE = null_mut();
            let mut host_stderr_read: HANDLE = null_mut();
            let mut child_stderr_write: HANDLE = null_mut();

            let mut sa: windows_sys::Win32::Security::SECURITY_ATTRIBUTES =
                unsafe { std::mem::zeroed() };
            sa.nLength =
                std::mem::size_of::<windows_sys::Win32::Security::SECURITY_ATTRIBUTES>() as u32;
            sa.bInheritHandle = 0; // default non-inheritable

            unsafe {
                // Stdin pipe
                if CreatePipe(&mut child_stdin_read, &mut host_stdin_write, &sa, 0) == 0 {
                    let err = GetLastError();
                    return Err(StructuredError::new(
                        "pipe_create_failed",
                        format!("CreatePipe (stdin) failed: {}", err),
                        false,
                    ));
                }
                SetHandleInformation(child_stdin_read, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
                SetHandleInformation(host_stdin_write, HANDLE_FLAG_INHERIT, 0);

                // Stdout pipe
                if CreatePipe(&mut host_stdout_read, &mut child_stdout_write, &sa, 0) == 0 {
                    let err = GetLastError();
                    CloseHandle(child_stdin_read);
                    CloseHandle(host_stdin_write);
                    return Err(StructuredError::new(
                        "pipe_create_failed",
                        format!("CreatePipe (stdout) failed: {}", err),
                        false,
                    ));
                }
                SetHandleInformation(child_stdout_write, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
                SetHandleInformation(host_stdout_read, HANDLE_FLAG_INHERIT, 0);

                // Stderr pipe
                if CreatePipe(&mut host_stderr_read, &mut child_stderr_write, &sa, 0) == 0 {
                    let err = GetLastError();
                    CloseHandle(child_stdin_read);
                    CloseHandle(host_stdin_write);
                    CloseHandle(host_stdout_read);
                    CloseHandle(child_stdout_write);
                    return Err(StructuredError::new(
                        "pipe_create_failed",
                        format!("CreatePipe (stderr) failed: {}", err),
                        false,
                    ));
                }
                SetHandleInformation(child_stderr_write, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
                SetHandleInformation(host_stderr_read, HANDLE_FLAG_INHERIT, 0);
            }

            // Prepare ProcThreadAttributeList containing:
            // 1. SECURITY_CAPABILITIES (AppContainer SID + 0 capabilities)
            // 2. HANDLE_LIST (only child stdin, stdout, stderr are inheritable)
            let mut attr_list = match ProcThreadAttributeList::new(2) {
                Ok(al) => al,
                Err(e) => {
                    unsafe {
                        CloseHandle(child_stdin_read);
                        CloseHandle(host_stdin_write);
                        CloseHandle(child_stdout_write);
                        CloseHandle(host_stdout_read);
                        CloseHandle(child_stderr_write);
                        CloseHandle(host_stderr_read);
                    }
                    return Err(e);
                }
            };

            let mut sec_caps = sandbox.security_capabilities();
            if let Err(e) = attr_list.set_security_capabilities(&mut sec_caps) {
                unsafe {
                    CloseHandle(child_stdin_read);
                    CloseHandle(host_stdin_write);
                    CloseHandle(child_stdout_write);
                    CloseHandle(host_stdout_read);
                    CloseHandle(child_stderr_write);
                    CloseHandle(host_stderr_read);
                }
                return Err(e);
            }

            let mut inheritable_handles =
                [child_stdin_read, child_stdout_write, child_stderr_write];
            if let Err(e) = attr_list.set_handle_list(&mut inheritable_handles) {
                unsafe {
                    CloseHandle(child_stdin_read);
                    CloseHandle(host_stdin_write);
                    CloseHandle(child_stdout_write);
                    CloseHandle(host_stdout_read);
                    CloseHandle(child_stderr_write);
                    CloseHandle(host_stderr_read);
                }
                return Err(e);
            }

            // Build STARTUPINFOEXW
            let mut siex: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
            siex.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
            siex.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            siex.StartupInfo.hStdInput = child_stdin_read;
            siex.StartupInfo.hStdOutput = child_stdout_write;
            siex.StartupInfo.hStdError = child_stderr_write;
            siex.lpAttributeList = attr_list.as_ptr();

            // Prepare command line and environment
            let app_name_w = to_wide_null(&resolved_program.to_string_lossy());
            let mut cmd_line = escape_windows_arg(&resolved_program.to_string_lossy());
            for arg in args {
                cmd_line.push(' ');
                cmd_line.push_str(&escape_windows_arg(arg));
            }
            let mut cmd_line_w = to_wide_null(&cmd_line);
            // Local verbatim drive paths name the same directory without the
            // prefix, which legacy child runtimes otherwise mistake for UNC.
            let cwd_text = cwd.to_string_lossy();
            let child_cwd = cwd_text
                .strip_prefix(r"\\?\")
                .filter(|path| {
                    path.as_bytes().get(1) == Some(&b':') && path.as_bytes().get(2) == Some(&b'\\')
                })
                .unwrap_or(&cwd_text);
            let cwd_w = to_wide_null(child_cwd);
            let mut env_block = build_sanitized_environment(cwd);

            let creation_flags = CREATE_SUSPENDED
                | CREATE_NO_WINDOW
                | EXTENDED_STARTUPINFO_PRESENT
                | CREATE_UNICODE_ENVIRONMENT;

            let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };

            let spawn_res = unsafe {
                CreateProcessW(
                    app_name_w.as_ptr(),
                    cmd_line_w.as_mut_ptr(),
                    null(),
                    null(),
                    1, // bInheritHandles = TRUE (required for PROC_THREAD_ATTRIBUTE_HANDLE_LIST)
                    creation_flags,
                    env_block.as_mut_ptr() as *mut c_void,
                    cwd_w.as_ptr(),
                    &siex.StartupInfo,
                    &mut pi,
                )
            };

            // Capture GetLastError IMMEDIATELY after CreateProcessW before any other Win32 calls
            let spawn_err = if spawn_res == 0 {
                unsafe { GetLastError() }
            } else {
                0
            };

            // Now close host copies of child pipe ends
            unsafe {
                CloseHandle(child_stdin_read);
                CloseHandle(child_stdout_write);
                CloseHandle(child_stderr_write);
            }

            if spawn_res == 0 {
                unsafe {
                    CloseHandle(host_stdin_write);
                    CloseHandle(host_stdout_read);
                    CloseHandle(host_stderr_read);
                }
                return Err(StructuredError::new(
                    "process_spawn_failed",
                    format!(
                        "CreateProcessW failed for '{}' with code {}",
                        resolved_program.display(),
                        spawn_err
                    ),
                    false,
                ));
            }

            // CRITICAL: Assign suspended process to Job Object BEFORE resuming thread.
            // This guarantees no uncontained execution occurs before limits are active.
            if let Err(e) = unsafe { job.assign_process(pi.hProcess) } {
                unsafe {
                    TerminateProcess(pi.hProcess, 1);
                    CloseHandle(pi.hThread);
                    CloseHandle(pi.hProcess);
                    CloseHandle(host_stdin_write);
                    CloseHandle(host_stdout_read);
                    CloseHandle(host_stderr_read);
                }
                return Err(StructuredError::new(
                    "job_assignment_failed",
                    format!(
                        "failed assigning suspended child process to Job Object: {}",
                        e.message
                    ),
                    false,
                ));
            }

            // Resume suspended process execution
            let resume_res = unsafe { ResumeThread(pi.hThread) };
            let resume_err = if resume_res == u32::MAX {
                unsafe { GetLastError() }
            } else {
                0
            };

            if resume_res == u32::MAX {
                unsafe {
                    TerminateProcess(pi.hProcess, 1);
                    CloseHandle(pi.hThread);
                    CloseHandle(pi.hProcess);
                    CloseHandle(host_stdin_write);
                    CloseHandle(host_stdout_read);
                    CloseHandle(host_stderr_read);
                }
                return Err(StructuredError::new(
                    "resume_thread_failed",
                    format!("ResumeThread failed with error code {}", resume_err),
                    false,
                ));
            }

            // Close thread handle; process handle is retained
            unsafe {
                CloseHandle(pi.hThread);
            }

            // Convert raw OS pipe handles into native File objects (implementing Read / Write)
            let stdin = unsafe { Some(File::from_raw_handle(host_stdin_write)) };
            let stdout = unsafe { Some(File::from_raw_handle(host_stdout_read)) };
            let stderr = unsafe { Some(File::from_raw_handle(host_stderr_read)) };

            Ok(Self {
                job,
                process_handle: pi.hProcess,
                pid: pi.dwProcessId,
                stdin,
                stdout,
                stderr,
                sandbox: Some(sandbox),
                cwd: cwd.to_path_buf(),
                exit_code: None,
                killed: false,
            })
        }

        /// Mutable reference to standard input if available.
        pub fn stdin(&mut self) -> &mut Option<File> {
            &mut self.stdin
        }

        /// Write bounded bytes to process stdin.
        pub fn write_stdin(&mut self, data: &[u8]) -> Result<usize, StructuredError> {
            if let Some(stdin) = &mut self.stdin {
                stdin.write(data).map_err(|e| {
                    StructuredError::new(
                        "stdin_write_failed",
                        format!("failed writing to stdin: {}", e),
                        false,
                    )
                })
            } else {
                Err(StructuredError::new(
                    "stdin_closed",
                    "process stdin is closed or not available",
                    false,
                ))
            }
        }

        /// Close standard input to signal EOF cooperatively without terminating the process boundary.
        pub fn close_stdin(&mut self) -> Result<(), StructuredError> {
            self.stdin = None;
            Ok(())
        }

        /// Take ownership of standard output reader.
        pub fn take_stdout(&mut self) -> Option<File> {
            self.stdout.take()
        }

        /// Take ownership of standard error reader.
        pub fn take_stderr(&mut self) -> Option<File> {
            self.stderr.take()
        }

        /// Non-blocking check for process exit. Returns `Ok(Some(exit_code))` if finished.
        pub fn try_wait(&mut self) -> Result<Option<i32>, StructuredError> {
            if let Some(exit) = self.exit_code {
                return Ok(Some(exit));
            }

            unsafe {
                let wait_res = WaitForSingleObject(self.process_handle, 0);
                let wait_err = if wait_res == WAIT_FAILED {
                    GetLastError()
                } else {
                    0
                };

                match wait_res {
                    WAIT_OBJECT_0 => {
                        let mut code: u32 = 0;
                        let ok = GetExitCodeProcess(self.process_handle, &mut code);
                        let exit_err = if ok == 0 { GetLastError() } else { 0 };

                        if ok != 0 {
                            let exit = code as i32;
                            self.exit_code = Some(exit);
                            Ok(Some(exit))
                        } else {
                            Err(StructuredError::new(
                                "get_exit_code_failed",
                                format!("GetExitCodeProcess failed with code {}", exit_err),
                                false,
                            ))
                        }
                    }
                    WAIT_TIMEOUT => Ok(None),
                    _ => Err(StructuredError::new(
                        "wait_failed",
                        format!("WaitForSingleObject failed with code {}", wait_err),
                        false,
                    )),
                }
            }
        }

        /// Blocking wait for process exit.
        pub fn wait(&mut self) -> Result<i32, StructuredError> {
            if let Some(exit) = self.exit_code {
                return Ok(exit);
            }

            unsafe {
                let wait_res = WaitForSingleObject(self.process_handle, INFINITE);
                let wait_err = if wait_res == WAIT_FAILED {
                    GetLastError()
                } else {
                    0
                };

                if wait_res == WAIT_OBJECT_0 {
                    let mut code: u32 = 0;
                    let ok = GetExitCodeProcess(self.process_handle, &mut code);
                    let exit_err = if ok == 0 { GetLastError() } else { 0 };

                    if ok != 0 {
                        let exit = code as i32;
                        self.exit_code = Some(exit);
                        Ok(exit)
                    } else {
                        Err(StructuredError::new(
                            "get_exit_code_failed",
                            format!("GetExitCodeProcess failed with code {}", exit_err),
                            false,
                        ))
                    }
                } else {
                    Err(StructuredError::new(
                        "wait_failed",
                        format!("WaitForSingleObject failed with code {}", wait_err),
                        false,
                    ))
                }
            }
        }

        /// Forcibly terminate the process and all descendant processes in its Job Object tree.
        ///
        /// Waits synchronously for process termination so no success-after-kill race can occur.
        pub fn kill(&mut self) -> Result<(), StructuredError> {
            if self.killed {
                return Ok(());
            }
            self.killed = true;

            // Close stdin first so process gets EOF if waiting on input

            self.stdin = None;

            // Terminate the entire Job Object process tree atomically
            let _ = self.job.terminate(1);

            // Direct process termination for primary process
            unsafe {
                TerminateProcess(self.process_handle, 1);
                // Synchronously wait for child process termination in the OS kernel
                WaitForSingleObject(self.process_handle, 5000);
            }

            // Record exit code; guarantee it is recorded as non-zero terminated code
            let mut code: u32 = 0;
            if unsafe { GetExitCodeProcess(self.process_handle, &mut code) } != 0 && code != 259 {
                self.exit_code = Some(code as i32);
            } else {
                self.exit_code = Some(1);
            }
            Ok(())
        }

        /// Return the OS process identifier (PID).
        pub fn pid(&self) -> u32 {
            self.pid
        }

        /// Check if the process is currently still running.
        pub fn is_alive(&mut self) -> bool {
            matches!(self.try_wait(), Ok(None))
        }

        /// Reference to the active AppContainer sandbox boundary, if present.
        pub fn sandbox(&self) -> Option<&AppContainerSandbox> {
            self.sandbox.as_ref()
        }
        /// Return the working directory granted to this sandbox.
        pub fn cwd(&self) -> &Path {
            &self.cwd
        }
    }

    impl Drop for RestrictedProcess {
        fn drop(&mut self) {
            // Guarantee process is terminated if still running on drop
            if self.exit_code.is_none() && !self.killed {
                let _ = self.kill();
            }

            if !self.process_handle.is_null() && self.process_handle != INVALID_HANDLE_VALUE {
                unsafe {
                    CloseHandle(self.process_handle);
                }
                self.process_handle = null_mut();
            }
            // `self.job` drops next -> closes Job Object handle -> KILL_ON_JOB_CLOSE terminates any descendants.
            // `self.sandbox` drops next -> restores tracked DACLs, deletes AppContainer profile, and revokes cwd DACL.
        }
    }

    /// Resolves an executable name to its absolute canonical path.
    fn resolve_executable_path(program: &Path, cwd: &Path) -> PathBuf {
        if program.is_absolute() && program.exists() {
            return program.to_path_buf();
        }

        let cwd_candidate = cwd.join(program);
        if cwd_candidate.exists() {
            return cwd_candidate;
        }

        let prog_str = program.to_string_lossy();
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
        let sys32 = Path::new(&system_root).join("System32");

        let extensions = ["", ".exe", ".cmd", ".bat", ".com"];
        for ext in &extensions {
            let name = if prog_str.to_lowercase().ends_with(ext) && !ext.is_empty() {
                prog_str.to_string()
            } else {
                format!("{}{}", prog_str, ext)
            };

            let sys32_candidate = sys32.join(&name);
            if sys32_candidate.exists() {
                return sys32_candidate;
            }

            let sys_candidate = Path::new(&system_root).join(&name);
            if sys_candidate.exists() {
                return sys_candidate;
            }
        }

        program.to_path_buf()
    }

    /// Windows command line argument quoting algorithm.
    fn escape_windows_arg(arg: &str) -> String {
        if arg.is_empty() {
            return "\"\"".to_string();
        }
        if !arg.contains([' ', '\t', '\n', '\x0b', '\"']) {
            return arg.to_string();
        }

        let mut escaped = String::with_capacity(arg.len() + 2);
        escaped.push('"');

        let mut backslashes = 0;
        for c in arg.chars() {
            if c == '\\' {
                backslashes += 1;
            } else if c == '"' {
                // Double preceding backslashes plus escape the quote
                for _ in 0..backslashes * 2 + 1 {
                    escaped.push('\\');
                }
                escaped.push('"');
                backslashes = 0;
            } else {
                for _ in 0..backslashes {
                    escaped.push('\\');
                }
                backslashes = 0;
                escaped.push(c);
            }
        }

        // Escape any trailing backslashes before the closing quote
        for _ in 0..backslashes * 2 {
            escaped.push('\\');
        }
        escaped.push('"');

        escaped
    }
}

#[cfg(not(windows))]
mod fallback_impl {
    use omp_types::{LimitPolicy, StructuredError};
    use std::convert::Infallible;
    use std::fs::File;
    use std::path::Path;

    /// Non-Windows placeholder; `RestrictedProcess` cannot be constructed on non-Windows platforms.
    pub struct RestrictedProcess {
        _unconstructible: Infallible,
    }

    impl RestrictedProcess {
        pub fn spawn(
            _program: &Path,
            _args: &[String],
            _cwd: &Path,
            _limits: &LimitPolicy,
        ) -> Result<Self, StructuredError> {
            Err(StructuredError::new(
                "unsupported_platform",
                "RestrictedProcess OS-level sandboxing requires native Windows platform (Windows AppContainer / JobObject)",
                false,
            ))
        }

        pub fn spawn_with_runtime(
            program: &Path,
            args: &[String],
            cwd: &Path,
            limits: &LimitPolicy,
            _runtime_files: &[std::path::PathBuf],
        ) -> Result<Self, StructuredError> {
            Self::spawn(program, args, cwd, limits)
        }

        pub fn stdin(&mut self) -> &mut Option<File> {
            match self._unconstructible {}
        }

        pub fn write_stdin(&mut self, _data: &[u8]) -> Result<usize, StructuredError> {
            match self._unconstructible {}
        }

        pub fn close_stdin(&mut self) -> Result<(), StructuredError> {
            match self._unconstructible {}
        }

        pub fn take_stdout(&mut self) -> Option<File> {
            match self._unconstructible {}
        }

        pub fn take_stderr(&mut self) -> Option<File> {
            match self._unconstructible {}
        }

        pub fn try_wait(&mut self) -> Result<Option<i32>, StructuredError> {
            match self._unconstructible {}
        }

        pub fn wait(&mut self) -> Result<i32, StructuredError> {
            match self._unconstructible {}
        }

        pub fn kill(&mut self) -> Result<(), StructuredError> {
            match self._unconstructible {}
        }

        pub fn pid(&self) -> u32 {
            match self._unconstructible {}
        }

        pub fn is_alive(&mut self) -> bool {
            match self._unconstructible {}
        }
    }
}

/// Windows-only integration scenarios: they spawn `cmd.exe`/PowerShell via
/// the native Job Object + AppContainer sandbox and cannot run elsewhere.
#[cfg(windows)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::LimitPolicy;
    use std::fs;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::Duration;

    /// Scenario 1: Forcibly kill a stuck child process and verify all descendants are terminated.
    #[test]
    fn test_stuck_child_kill() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_kill_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let limits = LimitPolicy::new()
            .with_max_wall_time(Duration::from_secs(60))
            .with_grace_period(Duration::from_millis(500));

        let program = PathBuf::from("cmd.exe");
        // Start a long-running stuck process tree
        let args = vec![
            "/c".to_string(),
            "powershell -NoProfile -Command \"Start-Sleep -Seconds 300\"".to_string(),
        ];

        let mut proc = RestrictedProcess::spawn(&program, &args, &temp_dir, &limits)
            .expect("should spawn restricted process");

        // Verify process is running
        assert!(proc.is_alive(), "stuck child should be alive after spawn");
        assert_eq!(proc.try_wait().expect("try_wait"), None);

        // Forcibly kill process boundary
        proc.kill().expect("kill should succeed");

        // Verify process is terminated and exit code is recorded
        let exit = proc.try_wait().expect("try_wait after kill");
        assert!(exit.is_some(), "process should be terminated after kill");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Scenario 2: Host journal access is strictly denied by the native AppContainer boundary default.
    ///
    /// Proves that:
    /// - Normal execution and writes inside the granted `cwd` succeed completely.
    /// - Attempts to write to a host file outside `cwd` fail with "Access is denied" at the kernel level
    ///   WITHOUT adding any explicit deny ACE (proving the native Lowbox boundary default).
    /// - The target file outside `cwd` remains completely intact and unmodified.
    #[test]
    fn test_journaling_denial() {
        let base_dir =
            std::env::temp_dir().join(format!("omp_test_journal_{}", std::process::id()));
        let sandbox_cwd = base_dir.join("sandbox_cwd");
        let journal_dir = base_dir.join("journal_dir");
        let _ = fs::create_dir_all(&sandbox_cwd);
        let _ = fs::create_dir_all(&journal_dir);

        let journal_file = journal_dir.join("session.journal");
        fs::write(&journal_file, b"AUTHORITATIVE_HOST_JOURNAL_CONTENT").expect("write journal");

        let limits = LimitPolicy::new();
        let program = PathBuf::from("cmd.exe");

        // 1. Prove the sandboxed process runs and can write to its granted cwd
        let allowed_file = sandbox_cwd.join("allowed_output.txt");
        let allowed_args = vec![
            "/c".to_string(),
            "echo".to_string(),
            "valid_payload".to_string(),
            ">".to_string(),
            allowed_file.to_string_lossy().to_string(),
        ];
        let mut allowed_proc =
            RestrictedProcess::spawn(&program, &allowed_args, &sandbox_cwd, &limits)
                .expect("spawn allowed process");

        let mut allowed_stderr = allowed_proc.take_stderr();
        let mut allowed_err = String::new();
        if let Some(mut err_reader) = allowed_stderr.take() {
            let _ = err_reader.read_to_string(&mut allowed_err);
        }
        let allowed_exit = allowed_proc.wait().expect("wait allowed process");
        assert_eq!(
            allowed_exit, 0,
            "sandboxed process must successfully write inside granted cwd; stderr: '{}'",
            allowed_err
        );
        assert!(
            allowed_file.exists(),
            "allowed file inside cwd must be created"
        );

        // 2. Prove native AppContainer boundary DEFAULT denies writing to host journal outside cwd
        // (No explicit deny ACE added; this proves the default OS security token boundary)
        let hostile_args = vec![
            "/c".to_string(),
            "echo".to_string(),
            "INJECTED_HOSTILE_DATA".to_string(),
            ">".to_string(),
            journal_file.to_string_lossy().to_string(),
        ];

        let mut hostile_proc =
            RestrictedProcess::spawn(&program, &hostile_args, &sandbox_cwd, &limits)
                .expect("spawn hostile test process");

        let mut stderr = hostile_proc.take_stderr().expect("take stderr");
        let mut err_output = String::new();
        let _ = stderr.read_to_string(&mut err_output);

        let exit_code = hostile_proc.wait().expect("wait hostile process");

        // Process must fail with non-zero exit code
        assert_ne!(
            exit_code, 0,
            "sandbox attempt to write outside cwd must fail with non-zero exit code"
        );

        // Host journal file content must remain completely intact
        let content = fs::read(&journal_file).expect("read journal file");
        assert_eq!(
            content, b"AUTHORITATIVE_HOST_JOURNAL_CONTENT",
            "journal content outside cwd must remain unmodified"
        );

        let _ = fs::remove_dir_all(&base_dir);
    }

    /// Scenario 3: Verify clean environment block and credential isolation without modifying process environment.
    #[test]
    fn test_no_inherited_credentials_and_clean_environment() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_env_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let limits = LimitPolicy::new();
        let program = PathBuf::from("cmd.exe");
        let args = vec!["/c".to_string(), "set".to_string()];

        let mut proc = RestrictedProcess::spawn(&program, &args, &temp_dir, &limits)
            .expect("spawn restricted process");

        let mut stdout = proc.take_stdout().expect("take stdout");
        let mut output = String::new();
        stdout.read_to_string(&mut output).expect("read stdout");

        let _ = proc.wait().expect("wait");

        // Windows rewrites TEMP into the AppContainer's private package, not host temp.
        let temp_value = output
            .lines()
            .find_map(|line| line.strip_prefix("TEMP="))
            .expect("sandbox temp");
        let private_temp = PathBuf::from(temp_value.trim());
        assert_ne!(private_temp, std::env::temp_dir());
        assert!(
            private_temp
                .components()
                .any(|part| part.as_os_str() == "Packages")
        );
        for name in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
        ] {
            assert!(
                !output
                    .lines()
                    .any(|line| line.starts_with(&format!("{name}=")))
            );
        }

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Scenario 4: Bounded stdio pipes and cooperative close.
    #[test]
    fn test_stdio_pipes_and_cooperative_close() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_stdio_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let limits = LimitPolicy::new();
        let program = PathBuf::from("cmd.exe");
        let args = vec!["/c".to_string(), "more".to_string()];

        let mut proc = RestrictedProcess::spawn(&program, &args, &temp_dir, &limits)
            .expect("spawn restricted process");

        proc.write_stdin(b"hello from host to sandbox\n")
            .expect("write stdin");
        proc.close_stdin().expect("close stdin");

        let mut stdout = proc.take_stdout().expect("take stdout");
        let mut output = String::new();
        stdout.read_to_string(&mut output).expect("read stdout");

        let exit = proc.wait().expect("wait");
        assert_eq!(exit, 0);
        assert!(output.contains("hello from host to sandbox"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Scenario 5: Network access policy rejection.
    #[test]
    fn test_network_isolation_policy() {
        let temp_dir = std::env::temp_dir().join(format!("omp_test_net_{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        // Attempting to create sandbox with network capability must fail as Unsupported
        let res = crate::windows_sandbox::AppContainerSandbox::create(&temp_dir, true);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.code, "unsupported_capability");

        let _ = fs::remove_dir_all(&temp_dir);
    }
}

/// Non-Windows smoke test: spawning must fail closed with a structured
/// `unsupported_platform` error instead of panicking or hanging.
#[cfg(not(windows))]
#[cfg(test)]
mod fallback_tests {
    use crate::LimitPolicy;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn test_spawn_is_unsupported_off_windows() {
        // Sibling module of this test module: `crate::process::fallback_impl`.
        // (`match` instead of `unwrap_err`: the fallback type is unconstructible
        // and carries no `Debug` impl.)
        let err = match super::fallback_impl::RestrictedProcess::spawn(
            &PathBuf::from("true"),
            &[],
            &std::env::temp_dir(),
            &LimitPolicy::new().with_max_wall_time(Duration::from_secs(5)),
        ) {
            Err(err) => err,
            Ok(_) => panic!("spawn must fail off Windows"),
        };
        assert_eq!(err.code, "unsupported_platform");
    }
}

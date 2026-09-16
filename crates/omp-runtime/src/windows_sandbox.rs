//! Native Windows AppContainer / Lowbox and Job Object sandbox enforcement.
//!
//! Provides OS-level security boundaries on Windows 11:
//! - Windows AppContainer isolation with Lowbox security token.
//! - Explicit no-network capability by default (network marked as Unsupported).
//! - ACL-granted isolated `cwd`, with host filesystem and journal outside `cwd` denied at kernel level.
//! - Low Mandatory Integrity Label (S:(ML;OICI;NW;;;LW)) applied to isolated `cwd` ensuring
//!   AppContainer Lowbox processes can write without Mandatory Integrity Check denial.
//! - Process Job Object with `KILL_ON_JOB_CLOSE`, memory, CPU, process count, and UI restrictions.
//! - Suspended creation with immediate Job Object assignment before thread resume to eliminate
//!   uncontained spawn races.
//! - Handle inheritance strictly confined to stdio pipes (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`).
//! - Sanitized environment block with required AppContainer system variables (LOCALAPPDATA,
//!   SystemRoot, etc.) sorted in case-insensitive alphabetical order.
//! - Tracked DACL restore on external binaries: host filesystem permissions are never permanently altered.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use omp_types::{LimitPolicy, StructuredError};
use windows_sys::Win32::Foundation::{
    BOOL, CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, HANDLE, INVALID_HANDLE_VALUE, LocalFree,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, DENY_ACCESS, EXPLICIT_ACCESS_W,
    GRANT_ACCESS, GetNamedSecurityInfoW, NO_MULTIPLE_TRUSTEE, REVOKE_ACCESS, SDDL_REVISION_1,
    SE_FILE_OBJECT, SetEntriesInAclW, SetNamedSecurityInfoW, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN,
};
use windows_sys::Win32::Security::Isolation::{
    CreateAppContainerProfile, DeleteAppContainerProfile,
};
use windows_sys::Win32::Security::{
    ACL, CONTAINER_INHERIT_ACE, DACL_SECURITY_INFORMATION, FreeSid, GetSecurityDescriptorSacl,
    OBJECT_INHERIT_ACE, PSECURITY_DESCRIPTOR, PSID, SECURITY_CAPABILITIES,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_JOB_TIME, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    JOB_OBJECT_UILIMIT_DESKTOP, JOB_OBJECT_UILIMIT_DISPLAYSETTINGS, JOB_OBJECT_UILIMIT_EXITWINDOWS,
    JOB_OBJECT_UILIMIT_GLOBALATOMS, JOB_OBJECT_UILIMIT_HANDLES, JOB_OBJECT_UILIMIT_READCLIPBOARD,
    JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS, JOB_OBJECT_UILIMIT_WRITECLIPBOARD,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_BASIC_UI_RESTRICTIONS,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectBasicAccountingInformation,
    JobObjectBasicUIRestrictions, JobObjectExtendedLimitInformation, QueryInformationJobObject,
    SetInformationJobObject, TerminateJobObject,
};
use windows_sys::Win32::System::Threading::{
    DeleteProcThreadAttributeList, InitializeProcThreadAttributeList, UpdateProcThreadAttribute,
};

/// Global counter for entropy in AppContainer profile naming.
static APP_CONTAINER_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Attribute key: PROC_THREAD_ATTRIBUTE_HANDLE_LIST
/// Value = (2 & 0xFFFF) | 0x00020000 = 0x00020002
pub const PROC_THREAD_ATTRIBUTE_HANDLE_LIST: usize = 0x00020002;

/// Attribute key: PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES
/// Value = (9 & 0xFFFF) | 0x00020000 = 0x00020009
pub const PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES: usize = 0x00020009;

/// Standard generic access masks for file ACLs.
pub const FILE_ALL_ACCESS: u32 = 0x001F01FF;
pub const FILE_GENERIC_READ: u32 = 0x00120089;
pub const FILE_GENERIC_WRITE: u32 = 0x00120116;
pub const FILE_GENERIC_EXECUTE: u32 = 0x001200A0;

/// Mint an opaque, collision-resistant AppContainer profile identifier.
fn mint_appcontainer_profile_name() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = now.as_nanos();
    let counter = APP_CONTAINER_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    format!("omp_{:016x}{:08x}{:08x}", nanos, pid, counter)
}

/// Tracks temporary DACL modifications and restores original permissions upon drop.
#[derive(Debug)]
pub struct TrackedDaclGrant {
    path: PathBuf,
    sec_desc: PSECURITY_DESCRIPTOR,
    original_dacl: *mut ACL,
}

unsafe impl Send for TrackedDaclGrant {}

impl Drop for TrackedDaclGrant {
    fn drop(&mut self) {
        if !self.sec_desc.is_null() {
            let path_str = self.path.to_string_lossy();
            let mut path_w = to_wide_null(&path_str);
            unsafe {
                let _ = SetNamedSecurityInfoW(
                    path_w.as_mut_ptr(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    null_mut(),
                    null_mut(),
                    self.original_dacl,
                    null_mut(),
                );
                LocalFree(self.sec_desc as *mut _);
            }
            self.sec_desc = null_mut();
            self.original_dacl = null_mut();
        }
    }
}

/// Saved mandatory integrity label restored when the sandbox drops.
#[derive(Debug)]
struct IntegrityLabelRestore {
    path: PathBuf,
    sec_desc: PSECURITY_DESCRIPTOR,
}

unsafe impl Send for IntegrityLabelRestore {}

impl IntegrityLabelRestore {
    fn capture(path: &Path) -> Self {
        let path_str = path.to_string_lossy();
        let path_w = to_wide_null(&path_str);
        let mut sacl: *mut ACL = null_mut();
        let mut sec_desc: PSECURITY_DESCRIPTOR = null_mut();
        const LABEL_SECURITY_INFORMATION: u32 = 0x00000010;
        let res = unsafe {
            GetNamedSecurityInfoW(
                path_w.as_ptr(),
                SE_FILE_OBJECT,
                LABEL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                null_mut(),
                &mut sacl,
                &mut sec_desc,
            )
        };
        if res != 0 || sec_desc.is_null() {
            if !sec_desc.is_null() {
                unsafe {
                    LocalFree(sec_desc as *mut _);
                }
            }
            Self {
                path: path.to_path_buf(),
                sec_desc: null_mut(),
            }
        } else {
            Self {
                path: path.to_path_buf(),
                sec_desc,
            }
        }
    }

    fn restore(&mut self) {
        let path_str = self.path.to_string_lossy();
        let mut path_w = to_wide_null(&path_str);
        const LABEL_SECURITY_INFORMATION: u32 = 0x00000010;
        if self.sec_desc.is_null() {
            let _ = set_integrity_sddl(&self.path, "S:(ML;OICI;NW;;;ME)");
            return;
        }
        unsafe {
            let mut sacl_present: BOOL = 0;
            let mut sacl: *mut ACL = null_mut();
            let mut sacl_defaulted: BOOL = 0;
            GetSecurityDescriptorSacl(
                self.sec_desc,
                &mut sacl_present,
                &mut sacl,
                &mut sacl_defaulted,
            );
            if sacl_present != 0 && !sacl.is_null() {
                let _ = SetNamedSecurityInfoW(
                    path_w.as_mut_ptr(),
                    SE_FILE_OBJECT,
                    LABEL_SECURITY_INFORMATION,
                    null_mut(),
                    null_mut(),
                    null_mut(),
                    sacl,
                );
            } else {
                let _ = set_integrity_sddl(&self.path, "S:(ML;OICI;NW;;;ME)");
            }
            LocalFree(self.sec_desc as *mut _);
            self.sec_desc = null_mut();
        }
    }
}

impl Drop for IntegrityLabelRestore {
    fn drop(&mut self) {
        self.restore();
    }
}

#[derive(Debug)]
pub struct AppContainerSandbox {
    profile_name: String,
    sid: PSID,
    cwd: PathBuf,
    network_allowed: bool,
    tracked_grants: Vec<TrackedDaclGrant>,
    label_restore: Option<IntegrityLabelRestore>,
}

unsafe impl Send for AppContainerSandbox {}
unsafe impl Sync for AppContainerSandbox {}

impl AppContainerSandbox {
    /// Create a new AppContainer profile and grant permissions exclusively to `cwd`.
    ///
    /// By default, `allow_network` is false. If network access is requested, it is rejected
    /// as `Unsupported` rather than granting arbitrary host network access.
    ///
    /// Uses an opaque minted token and never derives/reuses existing profiles on collision.
    pub fn create(cwd: &Path, allow_network: bool) -> Result<Self, StructuredError> {
        if allow_network {
            return Err(StructuredError::new(
                "unsupported_capability",
                "network capability is unsupported in Windows sandbox; arbitrary network access refused",
                false,
            ));
        }

        let mut sid: PSID = null_mut();
        let mut profile_name = String::new();
        let mut created = false;

        // Attempt creation with an opaque minted name. On collision, retry with a new mint.
        // Never derive/reuse a previously existing profile.
        for _ in 0..3 {
            profile_name = mint_appcontainer_profile_name();
            let profile_name_w = to_wide_null(&profile_name);
            let display_name_w = to_wide_null(&format!("OMP Sandbox {}", profile_name));
            let description_w = to_wide_null("Oh My Pi 2 AppContainer Sandbox");

            let hr = unsafe {
                CreateAppContainerProfile(
                    profile_name_w.as_ptr(),
                    display_name_w.as_ptr(),
                    description_w.as_ptr(),
                    null(),
                    0,
                    &mut sid,
                )
            };

            if hr >= 0 && !sid.is_null() {
                created = true;
                break;
            }

            let win32_err = (hr & 0xFFFF) as u32;
            if win32_err == ERROR_ALREADY_EXISTS {
                // Collision: mint a fresh unique token and retry; never derive old profile
                continue;
            } else {
                return Err(StructuredError::new(
                    "appcontainer_creation_failed",
                    format!(
                        "CreateAppContainerProfile failed for '{}' with HRESULT 0x{:08X} (Win32: {})",
                        profile_name, hr, win32_err
                    ),
                    false,
                ));
            }
        }

        if !created || sid.is_null() {
            return Err(StructuredError::new(
                "appcontainer_creation_failed",
                "failed to create unique AppContainer profile after multiple attempts",
                false,
            ));
        }

        let label_restore = IntegrityLabelRestore::capture(cwd);
        if let Err(e) = grant_appcontainer_access(cwd, sid) {
            drop(label_restore);
            let profile_name_w = to_wide_null(&profile_name);
            unsafe {
                let _ = DeleteAppContainerProfile(profile_name_w.as_ptr());
                FreeSid(sid);
            }
            return Err(e);
        }

        Ok(Self {
            profile_name,
            sid,
            cwd: cwd.to_path_buf(),
            network_allowed: false,
            tracked_grants: Vec::new(),
            label_restore: Some(label_restore),
        })
    }

    /// Deny this AppContainer access to a host-state directory inside cwd (typically `.omp`).
    pub fn deny_isolated_host_state(&mut self, path: &Path) -> Result<(), StructuredError> {
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|error| {
                StructuredError::new(
                    "host_state_dir",
                    format!("Failed to create '{}': {error}", path.display()),
                    false,
                )
            })?;
        }
        deny_appcontainer_access(path, self.sid)
    }

    /// Return the raw AppContainer SID.
    pub fn sid(&self) -> PSID {
        self.sid
    }

    /// Return the profile name.
    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    /// Return the granted working directory.
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Return whether network is allowed (always false by default).
    pub fn network_allowed(&self) -> bool {
        self.network_allowed
    }

    /// Construct the `SECURITY_CAPABILITIES` structure for `CreateProcessW`.
    pub fn security_capabilities(&self) -> SECURITY_CAPABILITIES {
        SECURITY_CAPABILITIES {
            AppContainerSid: self.sid,
            Capabilities: null_mut(),
            CapabilityCount: 0,
            Reserved: 0,
        }
    }

    /// Grant read/execute access to an external executable with tracked restore on drop.
    ///
    /// System binaries in Windows/System32 already possess `ALL_APPLICATION_PACKAGES` read/execute
    /// by default, so they require no modification. Files within `cwd` inherit access from `cwd`.
    /// For external non-system binaries, this temporarily adds the AppContainer SID and restores
    /// the original DACL when the sandbox drops.
    pub fn grant_program_access(&mut self, program_path: &Path) -> Result<(), StructuredError> {
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
        let sys_path = Path::new(&system_root);

        if program_path.exists()
            && !program_path.starts_with(&self.cwd)
            && !program_path.starts_with(sys_path)
        {
            let grant = grant_tracked_appcontainer_access(
                program_path,
                self.sid,
                FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
            )?;
            self.tracked_grants.push(grant);
        }
        Ok(())
    }

    /// Verify that the journal path is not inside the granted cwd.
    pub fn verify_journal_access_restricted(
        &self,
        journal_path: &Path,
    ) -> Result<bool, StructuredError> {
        if journal_path.starts_with(&self.cwd) {
            return Err(StructuredError::new(
                "journal_inside_sandbox_cwd",
                format!(
                    "security invariant violated: journal path '{}' is inside sandbox cwd '{}'",
                    journal_path.display(),
                    self.cwd.display()
                ),
                false,
            ));
        }
        Ok(true)
    }
}

impl Drop for AppContainerSandbox {
    fn drop(&mut self) {
        self.tracked_grants.clear();

        if !self.sid.is_null() {
            let _ = revoke_appcontainer_access(&self.cwd, self.sid);
            let profile_name_w = to_wide_null(&self.profile_name);
            unsafe {
                let _ = DeleteAppContainerProfile(profile_name_w.as_ptr());
                FreeSid(self.sid);
            }
            self.sid = null_mut();
        }
        self.label_restore.take();
    }
}

/// Manages a Windows Job Object governing process containment and resource limits.
///
/// Configured with:
/// - `KILL_ON_JOB_CLOSE`: Automatically terminates all processes in the job if the host dies or closes the handle.
/// - `ACTIVE_PROCESS_LIMIT`: Bounds descendant process tree.
/// - Memory limits (`JOB_OBJECT_LIMIT_PROCESS_MEMORY` & `JOB_OBJECT_LIMIT_JOB_MEMORY`).
/// - CPU time limits (`JOB_OBJECT_LIMIT_JOB_TIME`).
/// - UI restrictions (clipboard access, window handles, desktop switches).
#[derive(Debug)]
pub struct JobObjectBoundary {
    handle: HANDLE,
}

unsafe impl Send for JobObjectBoundary {}
unsafe impl Sync for JobObjectBoundary {}

impl JobObjectBoundary {
    /// Create and configure a Job Object enforcing `LimitPolicy`.
    pub fn new(limits: &LimitPolicy) -> Result<Self, StructuredError> {
        let handle = unsafe { CreateJobObjectW(null(), null()) };
        let create_err = if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            unsafe { GetLastError() }
        } else {
            0
        };

        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return Err(StructuredError::new(
                "job_object_create_failed",
                format!("CreateJobObjectW failed with error code {}", create_err),
                false,
            ));
        }

        let mut ext_limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        let mut flags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;

        // Process count bound
        if limits.max_child_processes > 0 {
            flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            ext_limits.BasicLimitInformation.ActiveProcessLimit = limits.max_child_processes;
        }

        // Memory limits
        if limits.max_memory_bytes > 0 {
            flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY;
            ext_limits.ProcessMemoryLimit = limits.max_memory_bytes;
            ext_limits.JobMemoryLimit = limits.max_memory_bytes;
        }

        // CPU execution time budget
        if let Some(cpu_time) = limits.max_cpu_time {
            flags |= JOB_OBJECT_LIMIT_JOB_TIME;
            // 100-nanosecond intervals
            ext_limits.BasicLimitInformation.PerJobUserTimeLimit =
                (cpu_time.as_nanos() / 100) as i64;
        }

        ext_limits.BasicLimitInformation.LimitFlags = flags;

        unsafe {
            let ok = SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &ext_limits as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            let set_err = if ok == 0 { GetLastError() } else { 0 };

            if ok == 0 {
                CloseHandle(handle);
                return Err(StructuredError::new(
                    "job_object_set_limits_failed",
                    format!(
                        "SetInformationJobObject (extended limits) failed with code {}",
                        set_err
                    ),
                    false,
                ));
            }

            // Enforce UI restrictions: prevent desktop switching, clipboard manipulation, handle sniffing.
            let mut ui_restrictions: JOBOBJECT_BASIC_UI_RESTRICTIONS = std::mem::zeroed();
            ui_restrictions.UIRestrictionsClass = JOB_OBJECT_UILIMIT_HANDLES
                | JOB_OBJECT_UILIMIT_READCLIPBOARD
                | JOB_OBJECT_UILIMIT_WRITECLIPBOARD
                | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
                | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
                | JOB_OBJECT_UILIMIT_GLOBALATOMS
                | JOB_OBJECT_UILIMIT_DESKTOP
                | JOB_OBJECT_UILIMIT_EXITWINDOWS;

            let ok = SetInformationJobObject(
                handle,
                JobObjectBasicUIRestrictions,
                &ui_restrictions as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_UI_RESTRICTIONS>() as u32,
            );
            let ui_err = if ok == 0 { GetLastError() } else { 0 };

            if ok == 0 {
                CloseHandle(handle);
                return Err(StructuredError::new(
                    "job_object_set_ui_restrictions_failed",
                    format!(
                        "SetInformationJobObject (UI restrictions) failed with code {}",
                        ui_err
                    ),
                    false,
                ));
            }
        }

        Ok(Self { handle })
    }

    /// Assign a process handle to this Job Object.
    ///
    /// # Safety
    /// The caller must own a valid open `process_handle`; the raw handle is
    /// dereferenced by the kernel during assignment.
    pub unsafe fn assign_process(&self, process_handle: HANDLE) -> Result<(), StructuredError> {
        let ok = unsafe { AssignProcessToJobObject(self.handle, process_handle) };
        let err = if ok == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };

        if ok == 0 {
            return Err(StructuredError::new(
                "job_object_assign_failed",
                format!("AssignProcessToJobObject failed with error code {}", err),
                false,
            ));
        }
        Ok(())
    }

    /// Forcibly terminate all processes associated with this Job Object.
    pub fn terminate(&self, exit_code: u32) -> Result<(), StructuredError> {
        let ok = unsafe { TerminateJobObject(self.handle, exit_code) };
        let err = if ok == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };

        if ok == 0 {
            return Err(StructuredError::new(
                "job_object_terminate_failed",
                format!("TerminateJobObject failed with error code {}", err),
                false,
            ));
        }
        Ok(())
    }

    /// Query accounting information (process count, CPU time, etc.).
    pub fn query_accounting(
        &self,
    ) -> Result<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, StructuredError> {
        let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { std::mem::zeroed() };
        let mut return_len = 0u32;
        let ok = unsafe {
            QueryInformationJobObject(
                self.handle,
                JobObjectBasicAccountingInformation,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                &mut return_len,
            )
        };
        let err = if ok == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };

        if ok == 0 {
            return Err(StructuredError::new(
                "job_object_query_failed",
                format!("QueryInformationJobObject failed with code {}", err),
                false,
            ));
        }
        Ok(info)
    }

    /// Return the raw OS handle.
    pub fn handle(&self) -> HANDLE {
        self.handle
    }
}

impl Drop for JobObjectBoundary {
    fn drop(&mut self) {
        if !self.handle.is_null() && self.handle != INVALID_HANDLE_VALUE {
            unsafe {
                CloseHandle(self.handle);
            }
            self.handle = null_mut();
        }
    }
}

/// Helper for building `PROC_THREAD_ATTRIBUTE_LIST` structures for `CreateProcessW`.
pub struct ProcThreadAttributeList {
    buffer: Vec<u8>,
}

impl ProcThreadAttributeList {
    /// Initialize an attribute list with the specified capacity.
    pub fn new(attribute_count: u32) -> Result<Self, StructuredError> {
        let mut size = 0usize;
        unsafe {
            InitializeProcThreadAttributeList(null_mut(), attribute_count, 0, &mut size);
        }
        if size == 0 {
            return Err(StructuredError::new(
                "attribute_list_init_failed",
                "InitializeProcThreadAttributeList returned 0 required size",
                false,
            ));
        }

        let mut buffer = vec![0u8; size];
        let ok = unsafe {
            InitializeProcThreadAttributeList(
                buffer.as_mut_ptr() as *mut _,
                attribute_count,
                0,
                &mut size,
            )
        };
        let init_err = if ok == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };

        if ok == 0 {
            return Err(StructuredError::new(
                "attribute_list_init_failed",
                format!(
                    "InitializeProcThreadAttributeList failed with error code {}",
                    init_err
                ),
                false,
            ));
        }

        Ok(Self { buffer })
    }

    /// Set the `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES` attribute.
    pub fn set_security_capabilities(
        &mut self,
        caps: &mut SECURITY_CAPABILITIES,
    ) -> Result<(), StructuredError> {
        let ok = unsafe {
            UpdateProcThreadAttribute(
                self.buffer.as_mut_ptr() as *mut _,
                0,
                PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                caps as *mut _ as *mut c_void,
                std::mem::size_of::<SECURITY_CAPABILITIES>(),
                null_mut(),
                null_mut(),
            )
        };
        let err = if ok == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };

        if ok == 0 {
            return Err(StructuredError::new(
                "attribute_set_security_caps_failed",
                format!(
                    "UpdateProcThreadAttribute (security caps) failed with code {}",
                    err
                ),
                false,
            ));
        }
        Ok(())
    }

    /// Set the `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` attribute.
    pub fn set_handle_list(&mut self, handles: &mut [HANDLE]) -> Result<(), StructuredError> {
        let ok = unsafe {
            UpdateProcThreadAttribute(
                self.buffer.as_mut_ptr() as *mut _,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                handles.as_mut_ptr() as *mut c_void,
                std::mem::size_of_val(handles),
                null_mut(),
                null_mut(),
            )
        };
        let err = if ok == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };

        if ok == 0 {
            return Err(StructuredError::new(
                "attribute_set_handle_list_failed",
                format!(
                    "UpdateProcThreadAttribute (handle list) failed with code {}",
                    err
                ),
                false,
            ));
        }
        Ok(())
    }

    /// Return the raw pointer to pass to `STARTUPINFOEXW.lpAttributeList`.
    pub fn as_ptr(&mut self) -> *mut c_void {
        self.buffer.as_mut_ptr() as *mut c_void
    }
}

impl Drop for ProcThreadAttributeList {
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            unsafe {
                DeleteProcThreadAttributeList(self.buffer.as_mut_ptr() as *mut _);
            }
        }
    }
}

/// Construct a sanitized, credential-free environment block for an AppContainer.
///
/// Strips all host environment variables, tokens, API keys, and session data.
/// Retains only the essential Windows system parameters:
/// - `LOCALAPPDATA`: Mandatory for Windows AppContainers (`%LOCALAPPDATA%\Packages`).
///   Missing LOCALAPPDATA triggers Win32 error 203 (ERROR_ENVVAR_NOT_FOUND).
/// - `SystemRoot`, `SystemDrive`, `WINDIR`, `COMSPEC`, `PATHEXT`, `PATH`.
/// - `TEMP` and `TMP` explicitly redirected into the sandbox's isolated `cwd`.
///
/// Under Win32 `CreateProcessW` with `CREATE_UNICODE_ENVIRONMENT`, the environment block
/// MUST be strictly sorted in case-insensitive alphabetical order and double-null terminated.
pub fn build_sanitized_environment(cwd: &Path) -> Vec<u16> {
    let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
    let system_drive = std::env::var("SystemDrive").unwrap_or_else(|_| r"C:".to_string());
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| r"C:\Windows".to_string());
    let comspec =
        std::env::var("COMSPEC").unwrap_or_else(|_| format!(r"{}\System32\cmd.exe", system_root));
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| r".COM;.EXE;.BAT;.CMD".to_string());
    let local_app_data = std::env::var("LOCALAPPDATA")
        .unwrap_or_else(|_| format!(r"{}\AppData\Local", system_drive));
    let user_profile =
        std::env::var("USERPROFILE").unwrap_or_else(|_| format!(r"{}\Users\Default", system_drive));
    let cwd_str = cwd.to_string_lossy();

    // Strict system-only PATH
    let path = format!(
        r"{}\System32;{};{}\System32\Wbem;{}\System32\WindowsPowerShell\v1.0\",
        system_root, system_root, system_root, system_root
    );

    let mut vars: Vec<(String, String)> = vec![
        ("COMSPEC".to_string(), comspec),
        ("LOCALAPPDATA".to_string(), local_app_data),
        ("PATH".to_string(), path),
        ("PATHEXT".to_string(), pathext),
        ("SystemDrive".to_string(), system_drive),
        ("SystemRoot".to_string(), system_root),
        ("TEMP".to_string(), cwd_str.to_string()),
        ("TMP".to_string(), cwd_str.to_string()),
        ("USERPROFILE".to_string(), user_profile),
        ("WINDIR".to_string(), windir),
    ];

    // Win32 requirement: sorted case-insensitively alphabetically by variable name
    vars.sort_by_key(|a| a.0.to_uppercase());

    let mut env_block: Vec<u16> = Vec::new();
    for (k, v) in vars {
        let entry = format!("{}={}", k, v);
        env_block.extend(entry.encode_utf16());
        env_block.push(0); // Null terminator per variable
    }
    env_block.push(0); // Double null terminator for the block
    env_block
}

/// Convert a Rust string to a null-terminated UTF-16 wide string vector.
pub fn to_wide_null(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn set_integrity_sddl(path: &Path, sddl: &str) -> Result<(), StructuredError> {
    let path_str = path.to_string_lossy();
    let mut path_w = to_wide_null(&path_str);
    let sddl = to_wide_null(sddl);

    unsafe {
        let mut sec_desc: PSECURITY_DESCRIPTOR = null_mut();
        let mut desc_size: u32 = 0;
        let ok = ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            SDDL_REVISION_1,
            &mut sec_desc,
            &mut desc_size,
        );

        if ok == 0 {
            return Err(StructuredError::new(
                "integrity_label_convert_failed",
                format!(
                    "ConvertStringSecurityDescriptorToSecurityDescriptorW failed with code {}",
                    GetLastError()
                ),
                false,
            ));
        }

        let mut sacl_present: BOOL = 0;
        let mut sacl: *mut ACL = null_mut();
        let mut sacl_defaulted: BOOL = 0;

        GetSecurityDescriptorSacl(sec_desc, &mut sacl_present, &mut sacl, &mut sacl_defaulted);

        if sacl_present == 0 || sacl.is_null() {
            LocalFree(sec_desc as *mut _);
            return Err(StructuredError::new(
                "integrity_label_missing",
                "Mandatory-integrity SDDL did not produce a SACL",
                false,
            ));
        }

        const LABEL_SECURITY_INFORMATION: u32 = 0x00000010;
        let apply = SetNamedSecurityInfoW(
            path_w.as_mut_ptr(),
            SE_FILE_OBJECT,
            LABEL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            null_mut(),
            sacl,
        );

        LocalFree(sec_desc as *mut _);

        if apply != 0 {
            return Err(StructuredError::new(
                "integrity_label_apply_failed",
                format!(
                    "SetNamedSecurityInfoW (mandatory label) failed with code {}",
                    apply
                ),
                false,
            ));
        }
    }
    Ok(())
}

/// Apply a Low Mandatory Integrity Label (S:(ML;OICI;NW;;;LW)) to the path.
pub fn set_low_mandatory_label(path: &Path) -> Result<(), StructuredError> {
    set_integrity_sddl(path, "S:(ML;OICI;NW;;;LW)")
}

/// Grant the specified AppContainer SID full control over the target path via Windows DACL,
/// and apply a Low Mandatory Integrity Label so the sandboxed process has full write capability.
pub fn grant_appcontainer_access(path: &Path, sid: PSID) -> Result<(), StructuredError> {
    modify_appcontainer_dacl(path, sid, GRANT_ACCESS, FILE_ALL_ACCESS)?;
    if let Err(error) = set_low_mandatory_label(path) {
        // Roll back the DACL grant so a partially-failed sandbox never
        // leaves the AppContainer SID with lingering access to the cwd.
        let _ = revoke_appcontainer_access(path, sid);
        return Err(error);
    }
    Ok(())
}

/// Revoke access of the specified AppContainer SID from the target path.
pub fn revoke_appcontainer_access(path: &Path, sid: PSID) -> Result<(), StructuredError> {
    modify_appcontainer_dacl(path, sid, REVOKE_ACCESS, 0)
}

/// Explicitly add a DENY ACE for the AppContainer SID to a target file or directory.
pub fn deny_appcontainer_access(path: &Path, sid: PSID) -> Result<(), StructuredError> {
    modify_appcontainer_dacl(path, sid, DENY_ACCESS, FILE_ALL_ACCESS)
}

/// Temporarily grant access to a path with automatic DACL restoration on drop.
pub fn grant_tracked_appcontainer_access(
    path: &Path,
    sid: PSID,
    permissions: u32,
) -> Result<TrackedDaclGrant, StructuredError> {
    let path_str = path.to_string_lossy();
    let mut path_w = to_wide_null(&path_str);

    unsafe {
        let mut old_dacl: *mut ACL = null_mut();
        let mut sec_desc: PSECURITY_DESCRIPTOR = null_mut();

        let res = GetNamedSecurityInfoW(
            path_w.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            &mut old_dacl,
            null_mut(),
            &mut sec_desc,
        );

        if res != 0 {
            return Err(StructuredError::new(
                "dacl_query_failed",
                format!(
                    "GetNamedSecurityInfoW for '{}' failed with code {}",
                    path.display(),
                    res
                ),
                false,
            ));
        }

        let mut ea: EXPLICIT_ACCESS_W = std::mem::zeroed();
        ea.grfAccessPermissions = permissions;
        ea.grfAccessMode = GRANT_ACCESS;
        ea.grfInheritance = CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE;
        ea.Trustee.pMultipleTrustee = null_mut();
        ea.Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
        ea.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN;
        ea.Trustee.ptstrName = sid as *mut u16;

        let mut new_dacl: *mut ACL = null_mut();
        let set_res = SetEntriesInAclW(1, &ea, old_dacl, &mut new_dacl);
        if set_res != 0 {
            LocalFree(sec_desc as *mut _);
            return Err(StructuredError::new(
                "dacl_set_entries_failed",
                format!("SetEntriesInAclW failed with code {}", set_res),
                false,
            ));
        }

        let apply_res = SetNamedSecurityInfoW(
            path_w.as_mut_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            new_dacl,
            null_mut(),
        );

        if !new_dacl.is_null() {
            LocalFree(new_dacl as *mut _);
        }

        if apply_res != 0 {
            LocalFree(sec_desc as *mut _);
            return Err(StructuredError::new(
                "dacl_apply_failed",
                format!(
                    "SetNamedSecurityInfoW for '{}' failed with code {}",
                    path.display(),
                    apply_res
                ),
                false,
            ));
        }

        Ok(TrackedDaclGrant {
            path: path.to_path_buf(),
            sec_desc,
            original_dacl: old_dacl,
        })
    }
}

/// Helper function to manipulate DACLs using Win32 Security APIs.
fn modify_appcontainer_dacl(
    path: &Path,
    sid: PSID,
    mode: i32,
    permissions: u32,
) -> Result<(), StructuredError> {
    let path_str = path.to_string_lossy();
    let mut path_w = to_wide_null(&path_str);

    unsafe {
        let mut old_dacl: *mut ACL = null_mut();
        let mut sec_desc: PSECURITY_DESCRIPTOR = null_mut();

        // Retrieve the current DACL
        let res = GetNamedSecurityInfoW(
            path_w.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            &mut old_dacl,
            null_mut(),
            &mut sec_desc,
        );

        if res != 0 {
            return Err(StructuredError::new(
                "dacl_query_failed",
                format!(
                    "GetNamedSecurityInfoW for '{}' failed with code {}",
                    path.display(),
                    res
                ),
                false,
            ));
        }

        let mut ea: EXPLICIT_ACCESS_W = std::mem::zeroed();
        ea.grfAccessPermissions = permissions;
        ea.grfAccessMode = mode;
        ea.grfInheritance = CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE;
        ea.Trustee.pMultipleTrustee = null_mut();
        ea.Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
        ea.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN;
        ea.Trustee.ptstrName = sid as *mut u16;

        let mut new_dacl: *mut ACL = null_mut();
        let set_res = SetEntriesInAclW(1, &ea, old_dacl, &mut new_dacl);
        if set_res != 0 {
            if !sec_desc.is_null() {
                LocalFree(sec_desc as *mut _);
            }
            return Err(StructuredError::new(
                "dacl_set_entries_failed",
                format!("SetEntriesInAclW failed with code {}", set_res),
                false,
            ));
        }

        let apply_res = SetNamedSecurityInfoW(
            path_w.as_mut_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            new_dacl,
            null_mut(),
        );

        if !new_dacl.is_null() {
            LocalFree(new_dacl as *mut _);
        }
        if !sec_desc.is_null() {
            LocalFree(sec_desc as *mut _);
        }

        if apply_res != 0 {
            return Err(StructuredError::new(
                "dacl_apply_failed",
                format!(
                    "SetNamedSecurityInfoW for '{}' failed with code {}",
                    path.display(),
                    apply_res
                ),
                false,
            ));
        }
    }

    Ok(())
}

use crate::transcript::NativeRow;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalGeometry {
    pub width: u16,
    pub height: u16,
}

impl TerminalGeometry {
    pub const fn new(width: u16, height: u16) -> Self {
        Self { width, height }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TerminalError {
    #[error("terminal write failure: {0}")]
    WriteFailure(String),
    #[error("terminal device disconnected")]
    DeviceDisconnected,
    #[error("geometry out of bounds: {0}x{1}")]
    OutOfBounds(u16, u16),
}

pub trait TerminalBackend: Send + Sync {
    fn geometry(&self) -> TerminalGeometry;
    fn write_row(&mut self, row: &NativeRow) -> Result<(), TerminalError>;
    fn write_rows(&mut self, rows: &[NativeRow]) -> Result<usize, TerminalError> {
        let mut count = 0;
        for r in rows {
            self.write_row(r)?;
            count += 1;
        }
        Ok(count)
    }
    fn clear_screen(&mut self) -> Result<(), TerminalError>;
    fn flush(&mut self) -> Result<(), TerminalError>;
}

#[derive(Clone, Debug)]
pub struct VirtualTerminal {
    pub geometry: TerminalGeometry,
    pub screen: Vec<NativeRow>,
    pub clear_count: usize,
    pub flush_count: usize,
    pub fail_after_n_writes: Option<usize>,
    pub total_writes_attempted: usize,
}

impl VirtualTerminal {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            geometry: TerminalGeometry::new(width, height),
            screen: Vec::new(),
            clear_count: 0,
            flush_count: 0,
            fail_after_n_writes: None,
            total_writes_attempted: 0,
        }
    }

    pub fn set_fail_after(&mut self, count: usize) {
        self.fail_after_n_writes = Some(count);
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.geometry = TerminalGeometry::new(width, height);
    }

    pub fn rendered_text(&self) -> Vec<String> {
        self.screen.iter().map(|r| r.text.clone()).collect()
    }
    pub fn accepted_rows_count(&self) -> usize {
        self.screen.len()
    }
}

impl TerminalBackend for VirtualTerminal {
    fn geometry(&self) -> TerminalGeometry {
        self.geometry
    }

    fn write_row(&mut self, row: &NativeRow) -> Result<(), TerminalError> {
        self.total_writes_attempted += 1;
        if let Some(fail_limit) = self.fail_after_n_writes
            && self.total_writes_attempted > fail_limit {
                return Err(TerminalError::WriteFailure(format!(
                    "simulated write failure after {fail_limit} rows"
                )));
            }
        self.screen.push(row.clone());
        Ok(())
    }

    fn clear_screen(&mut self) -> Result<(), TerminalError> {
        self.clear_count += 1;
        self.screen.clear();
        Ok(())
    }

    fn flush(&mut self) -> Result<(), TerminalError> {
        self.flush_count += 1;
        Ok(())
    }
}

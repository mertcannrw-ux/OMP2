--------------------------- MODULE ElasticSlots ---------------------------
(***************************************************************************)
(* Elastic Speculative Slots: a formally verified rendering protocol for   *)
(* streaming concurrent output blocks through a bounded terminal viewport  *)
(* into append-only scrollback.                                            *)
(*                                                                         *)
(* Three decoupled layers, related by invariants:                          *)
(*   1. semantic block state   (phase/mode/want/final/emitted per block)   *)
(*   2. logical history ledger (`history`: width-independent, exactly-once)*)
(*   3. physical native rows   (`native`: width-rendered, source-tagged)   *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
  N,                \* number of block identities (blocks are 1..N, in commit order)
  H,                \* maximum viewport (live transcript) height, in rows
  MaxResizes,       \* bound on resize events (keeps the state space finite)
  MaxLive,          \* uncommitted-block count that constitutes "pressure"
  RowValues,        \* finite row alphabet (what a semantic line of output "is")
  SnapshotValues,   \* finite universe of block contents (sequences of rows)
  NoFinal,          \* sentinel "this block has no final snapshot yet"
  Placeholder,      \* synthetic viewport row shown for an empty slot
  Blank,            \* synthetic viewport row for unused screen space
  OverflowMarker    \* synthetic viewport row summarizing hidden older blocks

ASSUME
  /\ N \in Nat \ {0}                                 \* at least one block
  /\ H \in Nat \ {0}                                 \* viewport can be nonempty
  /\ MaxResizes \in Nat                              \* zero resizes is allowed
  /\ MaxLive \in Nat \ {0}                           \* pressure threshold >= 1
  /\ IsFiniteSet(RowValues)                          \* finite row alphabet
  /\ RowValues /= {}                                 \* ... and nonempty
  /\ IsFiniteSet(SnapshotValues)                     \* finite snapshot universe
  /\ SnapshotValues \subseteq Seq(RowValues)         \* snapshots are row sequences
  /\ <<>> \in SnapshotValues                         \* the empty snapshot exists
  /\ (\E snapshot \in SnapshotValues : Len(snapshot) = 1)  \* length-1 snapshot exists
  /\ (\E snapshot \in SnapshotValues : Len(snapshot) > 1)  \* longer snapshot exists too
  /\ NoFinal \notin SnapshotValues                   \* sentinel distinct from real data
  /\ Placeholder \notin RowValues                    \* synthetic rows are not confusable
  /\ Blank \notin RowValues                          \* ... with semantic rows
  /\ OverflowMarker \notin RowValues                 \* ... and are pairwise distinct
  /\ Placeholder /= Blank
  /\ Placeholder /= OverflowMarker
  /\ Blank /= OverflowMarker

Blocks == 1..N                                       \* the block identities
ModelRows == {"row-a", "row-b"}                      \* concrete row alphabet for TLC
ModelSnapshots ==                                    \* richer snapshot universe
  {<<>>,
   <<"row-a">>,
   <<"row-b">>,
   <<"row-a", "row-b">>,
   <<"row-b", "row-a">>,
   <<"row-a", "row-b", "row-a">>}
SmallModelSnapshots == {<<>>, <<"row-a">>, <<"row-a", "row-b">>}

WidthValues == {"Wide", "Narrow"}                    \* two-point abstraction of terminal width
ResizeModes == {"Preserve", "Append", "Rebuild"}     \* policy chosen at a width-changing resize
ReplayModes == {"None", "Append", "Rebuild"}         \* pending replay (None = no replay in flight)
BlockModes == {"Undeclared", "Mutable", "AppendOnly"} \* presentation contract, fixed at Create
Phases == {"Absent", "Queued", "Active", "Finalized", "Committed"} \* block lifecycle
StopReasons == {"Running", "Graceful", "Detach", "WriteFailure"} \* why host stopped
NativeSources == {"Append", "Retire", "Replay", "Resize", "FailedWrite", "Exit"}
CellRows == RowValues \cup {Placeholder, Blank, OverflowMarker}
Cells == [owner : 0..N, row : CellRows]
TaggedRows == [owner : Blocks, row : RowValues]
NativeRows == [source : NativeSources, owner : 0..N, row : CellRows, width : WidthValues]

SnapshotLengths == {Len(snapshot) : snapshot \in SnapshotValues}
MaxSnapshotLength ==
  CHOOSE maximum \in SnapshotLengths :
    \A length \in SnapshotLengths : length <= maximum
MaxFailureRows == 2 * N * MaxSnapshotLength

BlankCell == [owner |-> 0, row |-> Blank]
OverflowCell == [owner |-> 0, row |-> OverflowMarker]

-----------------------------------------------------------------------------
(* State variables *)
-----------------------------------------------------------------------------
VARIABLES
  c,              \* commit frontier: blocks 1..c are committed (retired)
  phase,          \* lifecycle phase per block
  mode,           \* Mutable / AppendOnly contract per block
  want,           \* current speculative snapshot per block
  final,          \* frozen final snapshot per block (NoFinal until finalized)
  emitted,        \* rows of head block already streamed into history
  alloc,          \* painted slot height per block (rows on screen now)
  target,         \* requested slot height per block (animation target)
  history,        \* logical ledger (layer 2)
  native,         \* physical scrollback of current epoch (layer 3)
  width, height,  \* current terminal geometry
  resizes,        \* how many resizes happened (bounded by MaxResizes)
  epoch,          \* display epoch; Rebuild resets native and bumps this
  replayMode,     \* pending replay policy (None / Append / Rebuild)
  replayCursor,   \* first committed block to replay (invariantly 1 while replaying)
  replayEnd,      \* last committed block to replay (= c at replay start)
  replayPartial,  \* how many stable head rows to replay
  replayPrepared, \* replay frame computed and cut fixed (gates scheduler)
  replayCut,      \* rows of replay frame that must scroll into native
  flush,          \* explicit "retire everything" request (never reset)
  shutdown,       \* graceful shutdown initiated
  running,        \* host still alive; every action requires it
  stopReason      \* why we stopped (Running while alive)

vars == <<c, phase, mode, want, final, emitted, alloc, target,
          history, native, width, height, resizes, epoch,
          replayMode, replayCursor, replayEnd, replayPartial,
          replayPrepared, replayCut,
          flush, shutdown, running, stopReason>>

Maximum(left, right) == IF left >= right THEN left ELSE right

-----------------------------------------------------------------------------
(* Width rendering: soft-wrap reflow abstraction *)
-----------------------------------------------------------------------------
RECURSIVE DoubleRows(_)
DoubleRows(snapshot) ==
  IF Len(snapshot) = 0 THEN <<>>
  ELSE <<Head(snapshot), Head(snapshot)>> \o DoubleRows(Tail(snapshot))

Render(snapshot, wx) == IF wx = "Wide" THEN snapshot ELSE DoubleRows(snapshot)

Tag(i, snapshot) ==
  [j \in 1..Len(snapshot) |-> [owner |-> i, row |-> snapshot[j]]]

SnapshotSlice(snapshot, lo, hi) ==
  IF lo > hi THEN <<>> ELSE SubSeq(snapshot, lo, hi)

TagSlice(i, snapshot, lo, hi) == Tag(i, SnapshotSlice(snapshot, lo, hi))

NativeTag(source, i, snapshot, wx) ==
  [j \in 1..Len(Render(snapshot, wx)) |->
    [source |-> source, owner |-> i,
     row |-> Render(snapshot, wx)[j], width |-> wx]]

NativeTagSlice(source, i, snapshot, lo, hi, wx) ==
  NativeTag(source, i, SnapshotSlice(snapshot, lo, hi), wx)

NativeCells(source, cells, wx) ==
  [j \in 1..Len(cells) |->
    [source |-> source, owner |-> cells[j].owner,
     row |-> cells[j].row, width |-> wx]]

PrefixOf(sequence, count) == [j \in 1..count |-> sequence[j]]

-----------------------------------------------------------------------------
(* Logical history ledger helpers *)
-----------------------------------------------------------------------------
RECURSIVE CommittedRows(_, _)
CommittedRows(k, finals) ==
  IF k = 0 THEN <<>>
  ELSE CommittedRows(k - 1, finals) \o Tag(k, finals[k])

RECURSIVE TaggedRange(_, _, _)
TaggedRange(lo, hi, finals) ==
  IF lo > hi THEN <<>>
  ELSE Tag(lo, finals[lo]) \o TaggedRange(lo + 1, hi, finals)

RECURSIVE NativeRange(_, _, _, _, _)
NativeRange(source, lo, hi, finals, wx) ==
  IF lo > hi THEN <<>>
  ELSE NativeTag(source, lo, finals[lo], wx)
       \o NativeRange(source, lo + 1, hi, finals, wx)

RetirementRows(lo, hi, finals, firstEmitted) ==
  IF lo > hi THEN <<>>
  ELSE TagSlice(lo, finals[lo], firstEmitted + 1, Len(finals[lo]))
       \o TaggedRange(lo + 1, hi, finals)

NativeRetirementRows(source, lo, hi, finals, firstEmitted, wx) ==
  IF lo > hi THEN <<>>
  ELSE NativeTagSlice(
         source,
         lo,
         finals[lo],
         firstEmitted + 1,
         Len(finals[lo]),
         wx
       )
       \o NativeRange(source, lo + 1, hi, finals, wx)

FinalizedRange(lo, hi) ==
  \A i \in lo..hi : phase[i] = "Finalized"

Unemitted(snapshot, i, emission) ==
  IF mode[i] = "AppendOnly"
  THEN SnapshotSlice(snapshot, emission[i] + 1, Len(snapshot))
  ELSE snapshot

-----------------------------------------------------------------------------
(* Live viewport geometry *)
-----------------------------------------------------------------------------
Presented(ph, finals, emission, i, wx) ==
  \/ ph[i] = "Active"
  \/ /\ ph[i] = "Finalized"
     /\ Len(Render(Unemitted(finals[i], i, emission), wx)) > 0

PresentedSet(ph, finals, emission, wx) ==
  {i \in Blocks : Presented(ph, finals, emission, i, wx)}

PresentedCount(ph, finals, emission, wx) ==
  Cardinality(PresentedSet(ph, finals, emission, wx))

Overflow(ph, finals, emission, wx, hx) ==
  PresentedCount(ph, finals, emission, wx) > hx

SummaryRows(ph, finals, emission, wx, hx) ==
  IF hx > 0 /\ Overflow(ph, finals, emission, wx, hx) THEN 1 ELSE 0

NewerPresented(ph, finals, emission, wx, i) ==
  Cardinality({
    j \in Blocks :
      j > i /\ Presented(ph, finals, emission, j, wx)
  })

VisiblePresented(ph, finals, emission, wx, hx, i) ==
  /\ Presented(ph, finals, emission, i, wx)
  /\ IF Overflow(ph, finals, emission, wx, hx)
     THEN /\ hx > 0
          /\ NewerPresented(ph, finals, emission, wx, i) < hx - 1
     ELSE TRUE

RECURSIVE AllocationTotal(_, _)
AllocationTotal(al, i) ==
  IF i > N THEN 0 ELSE al[i] + AllocationTotal(al, i + 1)

RECURSIVE ReservationTotal(_, _, _)
ReservationTotal(al, requested, i) ==
  IF i > N THEN 0
  ELSE Maximum(al[i], requested[i]) + ReservationTotal(al, requested, i + 1)

AllocationStateOK(al, requested, ph, finals, emission, wx, hx) ==
  /\ al \in [Blocks -> 0..H]
  /\ requested \in [Blocks -> 0..H]
  /\ \A i \in Blocks :
       IF VisiblePresented(ph, finals, emission, wx, hx, i)
       THEN IF ph[i] = "Active"
            THEN /\ al[i] \in 1..H
                 /\ requested[i] \in 1..H
            ELSE /\ al[i] \in 1..H
                 /\ requested[i] = al[i]
       ELSE /\ al[i] = 0
            /\ requested[i] = 0
  /\ ReservationTotal(al, requested, 1)
     + SummaryRows(ph, finals, emission, wx, hx) <= hx

CanonicalAllocation(ph, finals, emission, wx, hx) ==
  [i \in Blocks |->
    IF VisiblePresented(ph, finals, emission, wx, hx, i) THEN 1 ELSE 0]

SnapshotHeight(ph, wants, finals, i, wx) ==
  CASE ph[i] = "Active" ->
         Maximum(1, Len(Render(Unemitted(wants[i], i, emitted), wx)))
    [] ph[i] = "Queued" ->
         Maximum(1, Len(Render(Unemitted(wants[i], i, emitted), wx)))
    [] ph[i] = "Finalized" ->
         Len(Render(Unemitted(finals[i], i, emitted), wx))
    [] OTHER -> 0

RECURSIVE FullRows(_, _, _, _, _)
FullRows(ph, wants, finals, wx, i) ==
  IF i > N THEN 0
  ELSE SnapshotHeight(ph, wants, finals, i, wx)
       + FullRows(ph, wants, finals, wx, i + 1)

CreatedCount == Cardinality({i \in Blocks : phase[i] /= "Absent"})

PartialHeadExists ==
  /\ c < CreatedCount
  /\ mode[c + 1] = "AppendOnly"
  /\ phase[c + 1] \in {"Active", "Finalized"}
  /\ emitted[c + 1] > 0

PartialHeadRows ==
  IF PartialHeadExists
  THEN TagSlice(c + 1, want[c + 1], 1, emitted[c + 1])
  ELSE <<>>

RowPressure == FullRows(phase, want, final, width, 1) > height

Pressure ==
  \/ RowPressure
  \/ CreatedCount - c >= MaxLive

RetirementRequested == flush \/ Pressure
Replaying == replayMode /= "None"

PreviewSource(i) ==
  IF phase[i] = "Active"
  THEN Unemitted(want[i], i, emitted)
  ELSE Unemitted(final[i], i, emitted)

PreviewCell(i, snapshot) ==
  LET rendered == Render(snapshot, width) IN
  [owner |-> i,
   row |-> IF Len(rendered) = 0
           THEN Placeholder
           ELSE rendered[Len(rendered)]]

Repeat(value, count) == [j \in 1..count |-> value]

Slot(i, snapshot, allocation) == Repeat(PreviewCell(i, snapshot), allocation)

RECURSIVE PresentedCells(_)
PresentedCells(i) ==
  IF i > N THEN <<>>
  ELSE (IF alloc[i] = 0 THEN <<>> ELSE Slot(i, PreviewSource(i), alloc[i]))
       \o PresentedCells(i + 1)

Screen ==
  Repeat(
    BlankCell,
    height - AllocationTotal(alloc, 1) - SummaryRows(phase, final, emitted, width, height)
  )
  \o (IF SummaryRows(phase, final, emitted, width, height) = 1
      THEN <<OverflowCell>>
      ELSE <<>>)
  \o PresentedCells(1)

ReplayRows ==
  IF ~Replaying
  THEN <<>>
  ELSE NativeRange("Replay", replayCursor, replayEnd, final, width)
       \o (IF replayPartial = 0
           THEN <<>>
           ELSE NativeTagSlice(
                  "Replay",
                  replayEnd + 1,
                  want[replayEnd + 1],
                  1,
                  replayPartial,
                  width
                ))

ReplayRoom ==
  Cardinality({j \in 1..height : Screen[j] = BlankCell})

RequiredReplayCut ==
  IF Len(ReplayRows) > ReplayRoom THEN Len(ReplayRows) - ReplayRoom ELSE 0

PreparedReplayTail ==
  IF replayPrepared
  THEN SnapshotSlice(ReplayRows, replayCut + 1, Len(ReplayRows))
  ELSE <<>>

Prefix(left, right) ==
  /\ Len(left) <= Len(right)
  /\ \A j \in 1..Len(left) : left[j] = right[j]

NoEarlierQueued(i) == \A j \in 1..(i - 1) : phase[j] /= "Queued"

BridgeHeight(sampled, requested) ==
  IF sampled < requested THEN requested
  ELSE IF sampled > 2 /\ requested = 1 THEN 2
  ELSE requested

-----------------------------------------------------------------------------
(* Initial State *)
-----------------------------------------------------------------------------
Init ==
  /\ c = 0
  /\ phase = [i \in Blocks |-> "Absent"]
  /\ mode = [i \in Blocks |-> "Undeclared"]
  /\ want = [i \in Blocks |-> <<>>]
  /\ final = [i \in Blocks |-> NoFinal]
  /\ emitted = [i \in Blocks |-> 0]
  /\ alloc = [i \in Blocks |-> 0]
  /\ target = [i \in Blocks |-> 0]
  /\ history = <<>>
  /\ native = <<>>
  /\ width = "Wide"
  /\ height = H
  /\ resizes = 0
  /\ epoch = 0
  /\ replayMode = "None"
  /\ replayCursor = 0
  /\ replayEnd = 0
  /\ replayPartial = 0
  /\ replayPrepared = FALSE
  /\ replayCut = 0
  /\ flush = FALSE
  /\ shutdown = FALSE
  /\ running = TRUE
  /\ stopReason = "Running"

-----------------------------------------------------------------------------
(* Actions *)
-----------------------------------------------------------------------------
Create(declaration) ==
  /\ running
  /\ ~shutdown
  /\ CreatedCount < N
  /\ phase[CreatedCount + 1] = "Absent"
  /\ declaration \in {"Mutable", "AppendOnly"}
  /\ phase' = [phase EXCEPT ![CreatedCount + 1] = "Queued"]
  /\ mode' = [mode EXCEPT ![CreatedCount + 1] = declaration]
  /\ UNCHANGED <<c, want, final, emitted, alloc, target, history, native,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

Admit(i) ==
  /\ running
  /\ ~shutdown
  /\ phase[i] = "Queued"
  /\ NoEarlierQueued(i)
  /\ LET newPhase == [phase EXCEPT ![i] = "Active"]
         newAlloc == [alloc EXCEPT ![i] = 1]
         newTarget == [target EXCEPT ![i] = 1]
     IN /\ ~Overflow(newPhase, final, emitted, width, height)
        /\ AllocationStateOK(newAlloc, newTarget, newPhase, final, emitted, width, height)
        /\ phase' = newPhase
        /\ alloc' = newAlloc
        /\ target' = newTarget
  /\ UNCHANGED <<c, mode, want, final, emitted, history, native, width, height,
                 resizes, epoch, replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

Update(i, snapshot) ==
  /\ running
  /\ ~shutdown
  /\ phase[i] \in {"Queued", "Active"}
  /\ (mode[i] = "Mutable" \/ Prefix(want[i], snapshot))
  /\ snapshot /= want[i]
  /\ want' = [want EXCEPT ![i] = snapshot]
  /\ UNCHANGED <<c, phase, mode, final, emitted, alloc, target, history, native,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

RequestAllocation(newTarget) ==
  /\ running
  /\ ~shutdown
  /\ AllocationStateOK(alloc, newTarget, phase, final, emitted, width, height)
  /\ newTarget /= target
  /\ target' = newTarget
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, history, native,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

ApplyAllocation(i) ==
  /\ running
  /\ ~shutdown
  /\ phase[i] = "Active"
  /\ alloc[i] /= target[i]
  /\ LET nextHeight == BridgeHeight(alloc[i], target[i])
         newAlloc == [alloc EXCEPT ![i] = nextHeight]
     IN /\ AllocationStateOK(newAlloc, target, phase, final, emitted, width, height)
        /\ alloc' = newAlloc
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, target, history, native,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

FinalizeActive(i, snapshot) ==
  /\ running
  /\ ~shutdown
  /\ phase[i] = "Active"
  /\ (mode[i] = "Mutable" \/ Prefix(want[i], snapshot))
  /\ LET newPhase == [phase EXCEPT ![i] = "Finalized"]
         newFinal == [final EXCEPT ![i] = snapshot]
         newAlloc == CanonicalAllocation(newPhase, newFinal, emitted, width, height)
     IN /\ phase' = newPhase
        /\ want' = [want EXCEPT ![i] = snapshot]
        /\ final' = newFinal
        /\ alloc' = newAlloc
        /\ target' = newAlloc
  /\ UNCHANGED <<c, mode, emitted, history, native, width, height,
                 resizes, epoch, replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

FinalizeQueued(i, snapshot) ==
  /\ running
  /\ ~shutdown
  /\ phase[i] = "Queued"
  /\ (mode[i] = "Mutable" \/ Prefix(want[i], snapshot))
  /\ LET newPhase == [phase EXCEPT ![i] = "Finalized"]
         newWant == [want EXCEPT ![i] = snapshot]
         newFinal == [final EXCEPT ![i] = snapshot]
         newAlloc == CanonicalAllocation(newPhase, newFinal, emitted, width, height)
     IN /\ phase' = newPhase
        /\ want' = newWant
        /\ final' = newFinal
        /\ alloc' = newAlloc
        /\ target' = newAlloc
  /\ UNCHANGED <<c, mode, emitted, history, native, width, height,
                 resizes, epoch, replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

AppendStable ==
  /\ running
  /\ ~shutdown
  /\ ~Replaying
  /\ c < CreatedCount
  /\ mode[c + 1] = "AppendOnly"
  /\ phase[c + 1] \in {"Active", "Finalized"}
  /\ RowPressure
  /\ emitted[c + 1] < Len(want[c + 1])
  /\ LET next == emitted[c + 1] + 1
         newEmitted == [emitted EXCEPT ![c + 1] = next]
         newAlloc == CanonicalAllocation(phase, final, newEmitted, width, height)
     IN /\ history' = history \o TagSlice(c + 1, want[c + 1], next, next)
        /\ native' = native \o NativeTagSlice("Append", c + 1, want[c + 1], next, next, width)
        /\ emitted' = newEmitted
        /\ alloc' = newAlloc
        /\ target' = newAlloc
  /\ UNCHANGED <<c, phase, mode, want, final,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

CompleteAppendOnly ==
  /\ running
  /\ ~Replaying
  /\ c < CreatedCount
  /\ mode[c + 1] = "AppendOnly"
  /\ phase[c + 1] = "Finalized"
  /\ emitted[c + 1] = Len(final[c + 1])
  /\ LET newPhase == [phase EXCEPT ![c + 1] = "Committed"]
         newEmitted == [emitted EXCEPT ![c + 1] = 0]
         newAlloc == CanonicalAllocation(newPhase, final, newEmitted, width, height)
     IN /\ c' = c + 1
        /\ phase' = newPhase
        /\ emitted' = newEmitted
        /\ alloc' = newAlloc
        /\ target' = newAlloc
  /\ UNCHANGED <<mode, want, final, history, native, width, height,
                 resizes, epoch, replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

BeginFlush ==
  /\ running
  /\ ~flush
  /\ flush' = TRUE
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target,
                 history, native, width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 shutdown, running, stopReason>>

RetireSuccess(batchEnd) ==
  /\ running
  /\ ~Replaying
  /\ batchEnd \in (c + 1)..N
  /\ FinalizedRange(c + 1, batchEnd)
  /\ RetirementRequested
  /\ history' = history \o RetirementRows(c + 1, batchEnd, final, emitted[c + 1])
  /\ native' =
       native
       \o NativeRetirementRows(
            "Retire",
            c + 1,
            batchEnd,
            final,
            emitted[c + 1],
            width
          )
  /\ LET newPhase == [i \in Blocks |->
           IF i <= batchEnd THEN "Committed" ELSE phase[i]]
         newEmitted == [i \in Blocks |->
           IF i <= batchEnd THEN 0 ELSE emitted[i]]
         newAlloc == CanonicalAllocation(newPhase, final, newEmitted, width, height)
     IN /\ c' = batchEnd
        /\ phase' = newPhase
        /\ emitted' = newEmitted
        /\ alloc' = newAlloc
        /\ target' = newAlloc
  /\ UNCHANGED <<mode, want, final, width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown, running, stopReason>>

RetireFailure(batchEnd, count) ==
  /\ running
  /\ ~Replaying
  /\ batchEnd \in (c + 1)..N
  /\ FinalizedRange(c + 1, batchEnd)
  /\ RetirementRequested
  /\ LET rows ==
           NativeRetirementRows(
             "FailedWrite",
             c + 1,
             batchEnd,
             final,
             emitted[c + 1],
             width
           )
     IN /\ count \in 0..Len(rows)
        /\ native' = native \o PrefixOf(rows, count)
  /\ running' = FALSE
  /\ stopReason' = "WriteFailure"
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target, history,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut, flush, shutdown>>

Resize(newWidth, newHeight, resizePolicy, pushed) ==
  /\ running
  /\ ~shutdown
  /\ resizes < MaxResizes
  /\ newWidth \in WidthValues
  /\ newHeight \in 0..H
  /\ resizePolicy \in ResizeModes
  /\ newWidth /= width \/ newHeight /= height
  /\ pushed \in 0..Len(Screen)
  /\ LET widthChanged == newWidth /= width
         effectiveMode == IF widthChanged THEN resizePolicy ELSE "Preserve"
         pushedRows == NativeCells("Resize", PrefixOf(Screen, pushed), width)
         beginReplay == effectiveMode /= "Preserve" /\ (c > 0 \/ PartialHeadExists)
         newPhase == phase
         newAlloc == CanonicalAllocation(newPhase, final, emitted, newWidth, newHeight)
     IN /\ width' = newWidth
        /\ height' = newHeight
        /\ resizes' = resizes + 1
        /\ alloc' = newAlloc
        /\ target' = newAlloc
        /\ native' = IF effectiveMode = "Rebuild"
                      THEN <<>>
                      ELSE native \o pushedRows
        /\ epoch' = IF effectiveMode = "Rebuild" THEN epoch + 1 ELSE epoch
        /\ replayMode' =
             IF beginReplay THEN effectiveMode
             ELSE IF Replaying THEN replayMode ELSE "None"
        /\ replayCursor' =
             IF beginReplay THEN 1
             ELSE IF Replaying THEN replayCursor ELSE 0
        /\ replayEnd' =
             IF beginReplay THEN c
             ELSE IF Replaying THEN replayEnd ELSE 0
        /\ replayPartial' =
             IF beginReplay
             THEN IF PartialHeadExists THEN emitted[c + 1] ELSE 0
             ELSE IF Replaying THEN replayPartial ELSE 0
        /\ replayPrepared' = FALSE
        /\ replayCut' = 0
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, history,
                 flush, shutdown, running, stopReason>>

PrepareReplay ==
  /\ running
  /\ Replaying
  /\ ~replayPrepared
  /\ replayPrepared' = TRUE
  /\ replayCut' = RequiredReplayCut
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target,
                 history, native, width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 flush, shutdown, running, stopReason>>

ReplaySynchronousSuccess ==
  /\ running
  /\ Replaying
  /\ replayPrepared
  /\ native' = native \o PrefixOf(ReplayRows, replayCut)
  /\ replayMode' = "None"
  /\ replayCursor' = 0
  /\ replayEnd' = 0
  /\ replayPartial' = 0
  /\ replayPrepared' = FALSE
  /\ replayCut' = 0
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target,
                 history, width, height, resizes, epoch,
                 flush, shutdown, running, stopReason>>

ReplaySynchronousFailure(count) ==
  /\ running
  /\ Replaying
  /\ replayPrepared
  /\ count \in 0..replayCut
  /\ native' = native \o PrefixOf(ReplayRows, count)
  /\ running' = FALSE
  /\ stopReason' = "WriteFailure"
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target, history,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 flush, shutdown>>

BeginGracefulShutdown ==
  /\ running
  /\ ~shutdown
  /\ LET newPhase == [i \in Blocks |->
           IF phase[i] = "Absent" THEN "Absent"
           ELSE IF i <= c THEN "Committed" ELSE "Finalized"]
         newFinal == [i \in Blocks |->
           IF phase[i] = "Absent" THEN NoFinal
           ELSE IF i <= c \/ phase[i] = "Finalized"
           THEN final[i]
           ELSE want[i]]
         newAlloc == CanonicalAllocation(newPhase, newFinal, emitted, width, height)
     IN /\ phase' = newPhase
        /\ final' = newFinal
        /\ alloc' = newAlloc
        /\ target' = newAlloc
  /\ flush' = TRUE
  /\ shutdown' = TRUE
  /\ UNCHANGED <<c, mode, want, emitted, history, native, width, height,
                 resizes, epoch, replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut,
                 running, stopReason>>

GracefulExit(push) ==
  /\ running
  /\ shutdown
  /\ ~Replaying
  /\ c = CreatedCount
  /\ push \in 0..1
  /\ push = 0 \/ height > 0
  /\ running' = FALSE
  /\ stopReason' = "Graceful"
  /\ native' = IF push = 0
               THEN native
               ELSE native \o NativeCells("Exit", <<Screen[1]>>, width)
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target, history,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut, flush, shutdown>>

DetachExit(push) ==
  /\ running
  /\ ~shutdown
  /\ push \in 0..1
  /\ push = 0 \/ height > 0
  /\ running' = FALSE
  /\ stopReason' = "Detach"
  /\ native' = IF push = 0
               THEN native
               ELSE native \o NativeCells("Exit", <<Screen[1]>>, width)
  /\ UNCHANGED <<c, phase, mode, want, final, emitted, alloc, target, history,
                 width, height, resizes, epoch,
                 replayMode, replayCursor, replayEnd, replayPartial,
                 replayPrepared, replayCut, flush, shutdown>>

-----------------------------------------------------------------------------
(* Existentially closed action wrappers *)
-----------------------------------------------------------------------------
RetireSuccessAction == \E batchEnd \in Blocks : RetireSuccess(batchEnd)

RetireFailureAction ==
  \E batchEnd \in Blocks :
    \E count \in 0..MaxFailureRows : RetireFailure(batchEnd, count)

ReplaySynchronousFailureAction ==
  \E count \in 0..MaxFailureRows : ReplaySynchronousFailure(count)

-----------------------------------------------------------------------------
(* Scheduler Gate & Next State Relation *)
-----------------------------------------------------------------------------
Next ==
  IF ~running THEN UNCHANGED vars ELSE
  IF replayPrepared
  THEN ReplaySynchronousSuccess \/ ReplaySynchronousFailureAction
  ELSE \/ \E declaration \in {"Mutable", "AppendOnly"} : Create(declaration)
       \/ \E i \in Blocks : Admit(i)
       \/ \E i \in Blocks, snapshot \in SnapshotValues : Update(i, snapshot)
       \/ \E newTarget \in [Blocks -> 0..H] : RequestAllocation(newTarget)
       \/ \E i \in Blocks : ApplyAllocation(i)
       \/ \E i \in Blocks, snapshot \in SnapshotValues : FinalizeActive(i, snapshot)
       \/ \E i \in Blocks, snapshot \in SnapshotValues : FinalizeQueued(i, snapshot)
       \/ AppendStable
       \/ CompleteAppendOnly
       \/ BeginFlush
       \/ RetireSuccessAction
       \/ RetireFailureAction
       \/ \E newWidth \in WidthValues, newHeight \in 0..H,
             resizePolicy \in ResizeModes, pushed \in 0..H :
              Resize(newWidth, newHeight, resizePolicy, pushed)
       \/ PrepareReplay
       \/ BeginGracefulShutdown
       \/ \E push \in 0..1 : GracefulExit(push)
       \/ \E push \in 0..1 : DetachExit(push)

Spec ==
  /\ Init
  /\ [][Next]_vars
  /\ WF_vars(RetireSuccessAction)
  /\ WF_vars(PrepareReplay)
  /\ WF_vars(ReplaySynchronousSuccess)
  /\ WF_vars(AppendStable)
  /\ WF_vars(CompleteAppendOnly)

-----------------------------------------------------------------------------
(* Safety Invariants *)
-----------------------------------------------------------------------------
TypeOK ==
  /\ c \in 0..N
  /\ phase \in [Blocks -> Phases]
  /\ mode \in [Blocks -> BlockModes]
  /\ want \in [Blocks -> SnapshotValues]
  /\ final \in [Blocks -> SnapshotValues \cup {NoFinal}]
  /\ emitted \in [Blocks -> 0..MaxSnapshotLength]
  /\ alloc \in [Blocks -> 0..H]
  /\ target \in [Blocks -> 0..H]
  /\ history \in Seq(TaggedRows)
  /\ native \in Seq(NativeRows)
  /\ width \in WidthValues
  /\ height \in 0..H
  /\ resizes \in 0..MaxResizes
  /\ epoch \in 0..MaxResizes
  /\ replayMode \in ReplayModes
  /\ replayCursor \in 0..(N + 1)
  /\ replayEnd \in 0..N
  /\ replayPartial \in 0..MaxSnapshotLength
  /\ replayPrepared \in BOOLEAN
  /\ replayCut \in 0..MaxFailureRows
  /\ flush \in BOOLEAN
  /\ shutdown \in BOOLEAN
  /\ running \in BOOLEAN
  /\ stopReason \in StopReasons

LifecycleShape ==
  /\ c <= CreatedCount
  /\ \A i \in 1..c :
       /\ phase[i] = "Committed"
       /\ mode[i] \in {"Mutable", "AppendOnly"}
  /\ \A i \in (c + 1)..CreatedCount :
       /\ phase[i] \in {"Queued", "Active", "Finalized"}
       /\ mode[i] \in {"Mutable", "AppendOnly"}
  /\ \A i \in (CreatedCount + 1)..N :
       /\ phase[i] = "Absent"
       /\ mode[i] = "Undeclared"

SnapshotDiscipline ==
  \A i \in Blocks :
    IF phase[i] \in {"Finalized", "Committed"}
    THEN /\ final[i] \in SnapshotValues
         /\ final[i] = want[i]
    ELSE final[i] = NoFinal

EmissionDiscipline ==
  /\ \A i \in Blocks :
       /\ emitted[i] <= Len(want[i])
       /\ (mode[i] /= "AppendOnly" => emitted[i] = 0)
       /\ (emitted[i] > 0 =>
            /\ i = c + 1
            /\ phase[i] \in {"Active", "Finalized"})
  /\ (PartialHeadExists => emitted[c + 1] <= Len(want[c + 1]))

Capacity == AllocationStateOK(alloc, target, phase, final, emitted, width, height)

ExactCommittedHistory == history = CommittedRows(c, final) \o PartialHeadRows

NoPrematureHistory ==
  \A j \in 1..Len(history) :
    LET owner == history[j].owner IN
    \/ /\ owner \in 1..c
       /\ phase[owner] = "Committed"
    \/ /\ PartialHeadExists
       /\ owner = c + 1

ScreenCapacity ==
  /\ Screen \in Seq(Cells)
  /\ Len(Screen) = height
  /\ \A i \in Blocks :
       Cardinality({j \in 1..height : Screen[j].owner = i}) = alloc[i]
  /\ Cardinality({j \in 1..height : Screen[j] = OverflowCell})
     = SummaryRows(phase, final, emitted, width, height)
  /\ Cardinality({j \in 1..height : Screen[j] = BlankCell})
     = height - AllocationTotal(alloc, 1)
       - SummaryRows(phase, final, emitted, width, height)

ReplayShape ==
  /\ (replayMode = "None" =>
       /\ replayCursor = 0
       /\ replayEnd = 0
       /\ replayPartial = 0
       /\ ~replayPrepared
       /\ replayCut = 0)
  /\ (replayMode /= "None" =>
       /\ replayCursor = 1
       /\ replayEnd \in 0..c
       /\ replayPartial <= MaxSnapshotLength
       /\ IF replayPrepared
          THEN /\ replayCut = RequiredReplayCut
               /\ Len(PreparedReplayTail) <= ReplayRoom
          ELSE replayCut = 0)

NativeSourceSafety ==
  \A j \in 1..Len(native) :
    LET owner == native[j].owner IN
    /\ (native[j].source = "Retire" =>
         /\ owner \in 1..c
         /\ phase[owner] = "Committed")
    /\ (native[j].source \in {"Append", "Replay"} =>
         /\ owner \in Blocks
         /\ (\/ owner \in 1..c
             \/ /\ owner = c + 1
                /\ mode[owner] = "AppendOnly"))
    /\ (native[j].source = "FailedWrite" => stopReason = "WriteFailure")
    /\ (native[j].source = "Exit" => ~running)

FailedWriteStops == stopReason = "WriteFailure" => ~running

-----------------------------------------------------------------------------
(* Temporal Properties *)
-----------------------------------------------------------------------------
HistoryExtension == Prefix(history, history')
HistoryMonotonicity == [][HistoryExtension]_vars

NativeEpochStep ==
  IF epoch' = epoch
  THEN Prefix(native, native')
  ELSE /\ epoch' = epoch + 1
       /\ native' = <<>>
NativeEpochDiscipline == [][NativeEpochStep]_vars

FinalsStayFixed ==
  \A i \in Blocks :
    phase[i] \in {"Finalized", "Committed"} => final'[i] = final[i]
FinalImmutability == [][FinalsStayFixed]_vars

AppendOnlyPrefixStep ==
  \A i \in Blocks :
    (mode[i] = "AppendOnly" /\ phase[i] \in {"Queued", "Active"})
    => Prefix(want[i], want'[i])
AppendOnlyMonotonicity == [][AppendOnlyPrefixStep]_vars

ResizeKeepsLogicalHistoryStep ==
  (width' /= width \/ height' /= height) =>
    /\ history' = history
    /\ c' = c
    /\ mode' = mode
    /\ want' = want
    /\ final' = final
    /\ emitted' = emitted
ResizeKeepsLogicalHistory == [][ResizeKeepsLogicalHistoryStep]_vars

StoppedStep == ~running => UNCHANGED vars
StoppedQuiescence == [][StoppedStep]_vars

AllFinalized ==
  \A i \in 1..CreatedCount : phase[i] \in {"Finalized", "Committed"}
AllCommitted ==
  /\ c = CreatedCount
  /\ history = CommittedRows(c, final)

FlushLiveness ==
  (AllFinalized /\ flush /\ shutdown /\ running /\ ~Replaying)
  ~> (AllCommitted \/ ~running)

ReplayLiveness == (Replaying /\ running) ~> (~Replaying \/ ~running)

QueuedDemand == \E i \in Blocks : phase[i] = "Queued"
QueuedPressureRetirement ==
  \A i \in Blocks :
    (/\ running
     /\ ~Replaying
     /\ c = i - 1
     /\ phase[i] = "Finalized"
     /\ Pressure
     /\ QueuedDemand)
    ~> (c >= i \/ ~running)

=============================================================================

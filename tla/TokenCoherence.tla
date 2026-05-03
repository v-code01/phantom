---------------------------- MODULE TokenCoherence ----------------------------
\* PHANTOM Token Coherence Protocol
\* Seeds from: arXiv 2603.15183 (Parakhin, March 2026)
\* Extensions: I2 (causal ordering) and I5 (atomic context injection) stubbed;
\*             proved in M3 and M1 respectively.

EXTENDS Naturals, FiniteSets

CONSTANTS
    Agents,     \* finite set of agent identifiers
    Artifacts,  \* finite set of artifact identifiers
    K,          \* staleness bound (positive natural number)
    None        \* sentinel value representing "no owner"; assigned in .cfg

ASSUME K \in Nat /\ K >= 1
ASSUME IsFiniteSet(Agents)    /\ Cardinality(Agents) >= 1
ASSUME IsFiniteSet(Artifacts) /\ Cardinality(Artifacts) >= 1

VARIABLES
    mstate,   \* [Artifacts -> {"M", "E", "S", "I"}]
    ver,      \* [Artifacts -> Nat]
    owner,    \* [Artifacts -> Agents \cup {None}]
    sharers,  \* [Artifacts -> SUBSET Agents]
    seen      \* [Agents -> [Artifacts -> Nat]]

vars == <<mstate, ver, owner, sharers, seen>>

\* State constraint: bound ver to keep TLC state space finite.
\* K+2 is sufficient to exercise KBound violations without infinite exploration.
VerBound == \A a \in Artifacts : ver[a] <= K + 2

TypeOK ==
    /\ mstate  \in [Artifacts -> {"M", "E", "S", "I"}]
    /\ ver     \in [Artifacts -> Nat]
    /\ owner   \in [Artifacts -> Agents \cup {None}]
    /\ sharers \in [Artifacts -> SUBSET Agents]
    /\ seen    \in [Agents -> [Artifacts -> Nat]]

Init ==
    /\ mstate  = [a \in Artifacts |-> "I"]
    /\ ver     = [a \in Artifacts |-> 0]
    /\ owner   = [a \in Artifacts |-> None]
    /\ sharers = [a \in Artifacts |-> {}]
    /\ seen    = [ag \in Agents |-> [a \in Artifacts |-> 0]]

Acquire(ag, a) ==
    /\ mstate[a] = "I"
    /\ mstate' = [mstate EXCEPT ![a] = "E"]
    /\ owner'  = [owner  EXCEPT ![a] = ag]
    /\ UNCHANGED <<ver, sharers, seen>>

Read(ag, a) ==
    /\ mstate[a] \in {"E", "S"}
    /\ mstate'  = [mstate  EXCEPT ![a] = "S"]
    /\ owner'   = [owner   EXCEPT ![a] = None]
    /\ sharers' = [sharers EXCEPT ![a] = sharers[a] \cup {ag}]
    /\ seen'    = [seen    EXCEPT ![ag][a] = ver[a]]
    /\ UNCHANGED <<ver>>

Write(ag, a) ==
    /\ mstate[a] = "E"
    /\ owner[a]  = ag
    /\ mstate' = [mstate EXCEPT ![a] = "M"]
    /\ ver'    = [ver    EXCEPT ![a] = ver[a] + 1]
    /\ seen'   = [seen   EXCEPT ![ag][a] = ver[a] + 1]
    /\ UNCHANGED <<owner, sharers>>

\* Writeback: M→E (retain exclusive ownership after write; not eviction).
\* Standard MESI evicts M→I; PHANTOM retains the cache line in E for reuse.
\* To evict, call Writeback then Invalidate.
Writeback(a) ==
    /\ mstate[a] = "M"
    /\ mstate' = [mstate EXCEPT ![a] = "E"]
    /\ UNCHANGED <<ver, owner, sharers, seen>>

Invalidate(a) ==
    /\ mstate[a] \in {"E", "S"}
    /\ mstate'  = [mstate  EXCEPT ![a] = "I"]
    /\ owner'   = [owner   EXCEPT ![a] = None]
    /\ sharers' = [sharers EXCEPT ![a] = {}]
    /\ UNCHANGED <<ver, seen>>

Next ==
    \/ \E ag \in Agents, a \in Artifacts : Acquire(ag, a)
    \/ \E ag \in Agents, a \in Artifacts : Read(ag, a)
    \/ \E ag \in Agents, a \in Artifacts : Write(ag, a)
    \/ \E a \in Artifacts : Writeback(a)
    \/ \E a \in Artifacts : Invalidate(a)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

SWMR ==
    \A a \in Artifacts :
        mstate[a] \in {"M", "E"} => sharers[a] = {}

\* SeenBound: each agent's last-seen version never exceeds the current version.
\* This is non-trivial (both ver and seen are Nat) and catches corruption bugs
\* where seen is written past ver, which would make KBound subtraction unsound.
SeenBound ==
    \A ag \in Agents, a \in Artifacts : seen[ag][a] <= ver[a]

KBound ==
    \A ag \in Agents, a \in Artifacts :
        ag \in sharers[a] => ver[a] - seen[ag][a] <= K

OwnerConsistency ==
    /\ \A a \in Artifacts :
        mstate[a] \in {"M", "E"} => owner[a] \in Agents
    /\ \A a \in Artifacts :
        mstate[a] \in {"I", "S"} => owner[a] = None

\* I2: Causal ordering — stub for M3 proof
\* Events within an agent's history must be causally ordered.
\* Full definition deferred to M3; TRUE here allows M0 TLC to pass.
I2 == TRUE

\* I5: Atomic context injection — stub for M1 proof
\* A context write must be visible to all agents atomically before the next step.
\* Full definition deferred to M1; TRUE here allows M0 TLC to pass.
I5 == TRUE

Invariants == SWMR /\ SeenBound /\ KBound /\ OwnerConsistency

=============================================================================

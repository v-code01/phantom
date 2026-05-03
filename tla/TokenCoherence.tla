---------------------------- MODULE TokenCoherence ----------------------------
\* PHANTOM Token Coherence Protocol
\* Seeds from: arXiv 2603.15183 (Parakhin, March 2026)
\* Invariants I4/I5 stubbed here; proved in M3.

EXTENDS Naturals, FiniteSets

CONSTANTS
    Agents,     \* finite set of agent identifiers
    Artifacts,  \* finite set of artifact identifiers
    K           \* staleness bound (positive natural number)

ASSUME K \in Nat /\ K >= 1
ASSUME IsFiniteSet(Agents)    /\ Cardinality(Agents) >= 1
ASSUME IsFiniteSet(Artifacts) /\ Cardinality(Artifacts) >= 1

None == CHOOSE x : x \notin Agents

VARIABLES
    mstate,   \* [Artifacts -> {"M", "E", "S", "I"}]
    ver,      \* [Artifacts -> Nat]
    owner,    \* [Artifacts -> Agents \cup {None}]
    sharers,  \* [Artifacts -> SUBSET Agents]
    seen      \* [Agents -> [Artifacts -> Nat]]

vars == <<mstate, ver, owner, sharers, seen>>

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
    /\ sharers' = [sharers EXCEPT ![a] = sharers[a] \cup {ag}]
    /\ seen'    = [seen    EXCEPT ![ag][a] = ver[a]]
    /\ UNCHANGED <<ver, owner>>

Write(ag, a) ==
    /\ mstate[a] = "E"
    /\ owner[a]  = ag
    /\ mstate' = [mstate EXCEPT ![a] = "M"]
    /\ ver'    = [ver    EXCEPT ![a] = ver[a] + 1]
    /\ seen'   = [seen   EXCEPT ![ag][a] = ver[a] + 1]
    /\ UNCHANGED <<owner, sharers>>

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

Spec == Init /\ [][Next]_vars

SWMR ==
    \A a \in Artifacts :
        mstate[a] \in {"M", "E"} => sharers[a] = {}

MonoVer ==
    \A a \in Artifacts : ver[a] >= 0

KBound ==
    \A ag \in Agents, a \in Artifacts :
        ag \in sharers[a] => ver[a] - seen[ag][a] <= K

OwnerConsistency ==
    /\ \A a \in Artifacts :
        mstate[a] \in {"M", "E"} => owner[a] \in Agents
    /\ \A a \in Artifacts :
        mstate[a] \in {"I", "S"} => owner[a] = None

Invariants == SWMR /\ MonoVer /\ KBound /\ OwnerConsistency

=============================================================================

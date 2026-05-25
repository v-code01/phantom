// I5 invariant: a block is only readable after all its bytes are written.
// This file is verified by `dafny verify dafny/KvInvariant.dfy`.

type byte = bv8
datatype Block = Block(data: seq<byte>, committed: bool)

// I5: every committed block in the slab has exactly `stride` bytes of data.
predicate I5(slab: seq<Block>, stride: nat) {
    forall i :: 0 <= i < |slab| ==>
        slab[i].committed ==> |slab[i].data| == stride
}

// I5 holds for the empty slab.
lemma I5Empty(stride: nat)
    ensures I5([], stride)
{}

// Writing `stride` bytes to block `id` and then committing preserves I5.
lemma CommitPreservesI5(
    slab: seq<Block>, id: nat, data: seq<byte>, stride: nat)
    requires I5(slab, stride)
    requires 0 <= id < |slab|
    requires |data| == stride
    ensures  I5(slab[id := Block(data, true)], stride)
{}

// Freeing a block (setting committed = false) preserves I5.
lemma FreePreservesI5(slab: seq<Block>, id: nat, stride: nat)
    requires I5(slab, stride)
    requires 0 <= id < |slab|
    ensures  I5(slab[id := Block(slab[id].data, false)], stride)
{}

// Allocating a freed slot does not change any data or committed flags;
// it only returns an index to the caller. I5 is trivially preserved.
// The precondition !slab[id].committed is the machine-checked encoding of
// the free-list contract: decref must reset committed=false before returning
// a slot to the free list, making alloc safe to hand the slot to a new writer.
lemma AllocPreservesI5(slab: seq<Block>, id: nat, stride: nat)
    requires I5(slab, stride)
    requires 0 <= id < |slab|
    requires !slab[id].committed
    ensures  I5(slab, stride)
{}

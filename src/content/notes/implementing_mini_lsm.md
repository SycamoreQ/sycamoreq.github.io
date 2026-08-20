---
title:  Implementing a Mini LSM in OCaml- Part 1 
date: 2026-08-20
description: This blog implements a MemTable in two ways.
tags: [Data Structures, OCaml]
coverImage: /cover_images/floppydisk-filepic.jpg
reference: https://npr.brightspotcdn.com/71/a1/2f24b5e44d058c2fd258117c1179/floppydisk-filepic.jpg
referenceText: "cover image: a random floppy disk"
draft: false
---

- Hello. Its been quite sometime since the last blog and that is because I was speed running progress through Axiom. I have also started to read and learn a lot about databases and database architectures. Specifically on MVCC, distributed databases, WAL and various DB related data structures which is the topic of this blog. I will be implementing a `Mini LSM` in OCaml. This will help me get better at OCaml as well as implement a very important piece of structure in databases that support many transaction level processing in databases.

- I will be following a specific Rust manual implementation of this, which essentially uses a `SkipList` to implement its `MemTable`. OCaml does not have a direct package for that so I have decided to first get my hands dirty and implement an Array based implementation following this article for the `MemTable`. After which I will take my time to implement the SkipList in OCaml, test it and then use that to implement the MemTable. 

- Due to sheer volume of things to be implemented and presented, I will divide this into a few parts and maybe implement more complex things that adds on top of this directly. So if you are interested watch out for the rest of the blogs. 

- All the code can be found here: [mini-lsm-ocaml](https://github.com/SycamoreQ/mini-lsm-ocaml)

## What is a Mini-LSM? What is a MemTable? 

An **LSM tree** (Log-Structured Merge tree) is a data structure built around a simple idea: writes are cheap if you never do them in place. Instead of seeking to some location on disk and mutating a byte, every write — insert, update, or delete — gets appended sequentially, first into an in-memory buffer and eventually into an immutable, sorted file on disk. Reads pay for this convenience by having to check multiple places (the in-memory buffer, then progressively older on-disk files), and the database periodically has to *compact* — merge several of these sorted files back into fewer, larger ones, discarding overwritten or deleted keys along the way. This is the design behind most modern write-heavy storage engines — RocksDB, LevelDB, Cassandra's storage layer, and most databases built on top of them use some variant of it. It trades read complexity for write throughput, which is exactly the tradeoff most OLTP and time-series workloads want.

A **MemTable** is the "in-memory buffer" half of that picture — the very first place a write lands, before it's durable on disk (and usually after also being appended to a write-ahead log, for crash recovery). Structurally, a MemTable has to do three things well:

- **Fast point lookups** — does this key exist, and what's its latest value?
- **Fast inserts and updates**, including recording deletes as *tombstones* rather than physically removing anything (actual removal happens later, during compaction)
- **In-order iteration** — because when the MemTable eventually fills up and gets flushed to disk, it needs to be written out as a *sorted* file (an SSTable — "Sorted String Table")

The canonical choice here — the one LevelDB, RocksDB, and the Rust `mini-lsm` tutorial I'm following all reach for — is a **skip list**: a linked structure with multiple "express lanes" of pointers that gives expected O(log n) search and insert, and, crucially, lets you insert without shifting anything else in memory, which matters a lot when the structure is being mutated constantly.

OCaml doesn't have a skip list in its standard library, and I didn't want to reach for a third-party one before building some intuition for the problem myself. So I decided to first implement the MemTable the "simple" way — a sorted structure of entries searched with binary search — get that correct, and *then* go build a skip list in OCaml on its own before swapping it in underneath the MemTable. That's the order this post follows: an array-backed attempt first, then the real skip list.

## Part 1: an array-backed MemTable

### The entry and MemTable types

```ocaml
type memtable_entry = {
  key : bytes;
  value : bytes option;
  timestamp : int64;
  deleted : bool;
}

type memtable = {
  mutable entries : memtable_entry array;
  mutable size : int;
}
```

`key` and `value` are `bytes` rather than, say, an `int8 list` — `bytes` is packed, gives O(1) indexed access, and gets lexicographic ordering for free via `compare`, which is exactly the ordering Rust's `Ord for [u8]` gives `&[u8]`. `value` is an `option` because an entry can be a **tombstone**: a deleted key still needs a live entry (with `value = None`) so a lookup correctly reports "deleted" instead of silently falling through to an older, stale value sitting in an on-disk SSTable underneath.

`entries` is an `array`, not a `list` — that was actually a correction I made partway through, and it's worth explaining why, because it's the whole reason binary search is worth doing in the first place. Binary search needs O(1) random access to the midpoint at every step. An OCaml `list` only gives O(1) access to its *head* — reaching the middle means walking `n/2` cons cells, which turns an O(log n) search into an O(n log n) one, worse than just scanning the list once. An `array` is the right structural analog to Rust's `Vec<T>` / slice here.

Both `entries` and `size` are marked `mutable`, since this structure gets mutated in place rather than rebuilt — every write either overwrites a slot in the existing array or replaces the whole array with a slightly bigger one, mirroring `&mut self` on the Rust side.

### Finding a key: `get_index`

The Rust reference implementation finds a key with `Vec::binary_search_by_key`, so the first thing I needed in OCaml was that function — and specifically *the same algorithm* Rust uses, not just any binary search. Rust's current `core::slice::binary_search_by` isn't the textbook version; it's a branchless variant (tracking a `base`/`half` pair instead of `left`/`right`/`mid`) written so the loop's iteration count depends only on the slice length, keeping it branch-predictor-friendly:

```ocaml
let binary_search_by (arr : 'a array) (f : 'a -> int) : (int, int) result =
  let len = Array.length arr in
  if len = 0 then Error 0
  else begin
    let size = ref len in
    let base = ref 0 in
    while !size > 1 do
      let half = !size / 2 in
      let mid = !base + half in
      let cmp = f arr.(mid) in
      base := if cmp > 0 then !base else mid;
      size := !size - half
    done;
    let cmp = f arr.(!base) in
    if cmp = 0 then Ok !base
    else Error (!base + if cmp < 0 then 1 else 0)
  end

let binary_search_by_key (arr : 'a array) (key : 'b) (f : 'a -> 'b) : (int, int) result =
  binary_search_by arr (fun x -> compare (f x) key)
```

`binary_search_by` takes a comparator (negative/zero/positive, same convention as `Stdlib.compare`) instead of Rust's `Ordering`. The return type is `('a, 'b) result` — `Stdlib`'s built-in `Ok`/`Error` variant lines up neatly with Rust's `Result<usize, usize>`: `Ok idx` means found at `idx`; `Error idx` means not found, and `idx` is exactly where the key would need to be inserted to keep `entries` sorted.

`get_index` plugs the MemTable's fields into that directly:

```ocaml
let get_index (t : memtable) (key : bytes) : (int, int) result =
  binary_search_by_key t.entries key (fun e -> e.key)
```

### Writing a key: `set`

```ocaml
let set (t : memtable) (key : bytes) (value : bytes) (timestamp : int64) : unit =
  let entry = { key; value = Some value; timestamp; deleted = false } in
  match get_index t key with
  | Ok idx ->
    (match t.entries.(idx).value with
     | Some v ->
       if Bytes.length value < Bytes.length v then
         t.size <- t.size - (Bytes.length v - Bytes.length value)
       else
         t.size <- t.size + (Bytes.length value - Bytes.length v)
     | None -> ());
    t.entries.(idx) <- entry
  | Error idx ->
    t.size <- t.size + Bytes.length key + Bytes.length value + 16 + 1;
    let len = Array.length t.entries in
    let new_entries = Array.make (len + 1) entry in
    Array.blit t.entries 0 new_entries 0 idx;
    Array.blit t.entries idx new_entries (idx + 1) (len - idx);
    t.entries <- new_entries
```

Two OCaml gotchas bit me while writing this, worth flagging for anyone else coming from Rust: `=` in OCaml is structural *equality*, not assignment — mutating a field needs `<-`, and only works at all once the field is declared `mutable` in the type. Neither is a compile error you get for free; both are things I had to go back and fix.

The `Ok idx` branch reconciles `size` against the byte-length difference between the old and new value before overwriting the slot. The `Error idx` branch is the more interesting one: since an OCaml `array` is fixed-length the moment it's allocated — unlike a `Vec`, which owns spare capacity — "inserting" a new key means allocating a new array one slot larger and copying the old contents around the gap. `Array.blit` copies everything before `idx` unchanged, then everything from `idx` onward shifted right by one, leaving `idx` free for the new entry.

That last part is also exactly the array's weak point. Every insert into a new key is an O(n) copy of the whole backing array, no matter where the key lands. A skip list doesn't have this problem — inserting a node only touches the handful of predecessor pointers around it — so that's the natural next thing to build.

## Part 2: a skip list

The `mini-lsm` tutorial I'm following doesn't hand-roll its own skip list — it uses [`crossbeam-skiplist`](https://docs.rs/crossbeam-skiplist/latest/crossbeam_skiplist/), a *lock-free concurrent* skip list built with CAS loops and epoch-based memory reclamation, so `put`/`get` only need a shared reference and multiple threads can hit the memtable at once without a mutex. Porting that concurrency machinery faithfully is a genuinely separate, much bigger project than what a first working MemTable needs, and OCaml doesn't have an off-the-shelf epoch-reclamation library to lean on. So I implemented the classical (Pugh) probabilistic skip list crossbeam's algorithm is itself built on top of: single-domain, imperative, unsynchronized — the same ordered-map interface (insert/get/iterate, upsert-on-insert, sorted order) the MemTable actually needs, without the lock-free layer on top. It's a reasonable foundation to make concurrent later.

```ocaml
(* skiplist.ml *)

type ('k, 'v) node = {
  key : 'k;
  mutable value : 'v;
  forward : ('k, 'v) node option array;  (* forward.(i) = next node at level i *)
}

type ('k, 'v) t = {
  compare : 'k -> 'k -> int;
  max_level : int;
  p : float;                            (* probability of promoting to the next level *)
  mutable level : int;                  (* highest level currently in use (>= 1) *)
  header : ('k, 'v) node option array;  (* header.(i) = first node at level i *)
  mutable size : int;
}

let create ?(max_level = 32) ?(p = 0.5) (compare : 'k -> 'k -> int) : ('k, 'v) t =
  { compare; max_level; p; level = 1; header = Array.make max_level None; size = 0 }

let length t = t.size
let is_empty t = t.size = 0

let random_level t =
  let lvl = ref 1 in
  while Random.float 1.0 < t.p && !lvl < t.max_level do
    incr lvl
  done;
  !lvl

(* Walks from the highest occupied level down to level 0, advancing as far
   right as possible at each level before dropping down. Returns the
   per-level predecessor array (`None` = header) — Pugh's "update" vector,
   used by both [insert] (to splice in a new node) and [get] (whose
   update.(0) forward pointer is the search candidate). *)
let find_predecessors t key =
  let update = Array.make t.max_level None in
  let current_forward = ref t.header in
  let current_node = ref None in
  for i = t.level - 1 downto 0 do
    let advancing = ref true in
    while !advancing do
      match !current_forward.(i) with
      | Some next_node when t.compare next_node.key key < 0 ->
        current_node := Some next_node;
        current_forward := next_node.forward
      | _ -> advancing := false
    done;
    update.(i) <- !current_node
  done;
  update

let candidate_after t update =
  match update.(0) with
  | Some pred -> pred.forward.(0)
  | None -> t.header.(0)

let get t key : 'v option =
  let update = find_predecessors t key in
  match candidate_after t update with
  | Some node when t.compare node.key key = 0 -> Some node.value
  | _ -> None

let contains_key t key = Option.is_some (get t key)

(* Insert [key, value]. If [key] already exists, overwrite its value in
   place — mini-lsm requires this so a single memtable never holds more
   than one entry per key. *)
let insert t key value : unit =
  let update = find_predecessors t key in
  match candidate_after t update with
  | Some node when t.compare node.key key = 0 -> node.value <- value
  | _ ->
    let lvl = random_level t in
    if lvl > t.level then begin
      for i = t.level to lvl - 1 do
        update.(i) <- None (* brand-new level: header is the only predecessor *)
      done;
      t.level <- lvl
    end;
    let forward = Array.make lvl None in
    let new_node = { key; value; forward } in
    for i = 0 to lvl - 1 do
      match update.(i) with
      | Some pred ->
        forward.(i) <- pred.forward.(i);
        pred.forward.(i) <- Some new_node
      | None ->
        forward.(i) <- t.header.(i);
        t.header.(i) <- Some new_node
    done;
    t.size <- t.size + 1

(* Visit every entry in ascending key order. The level-0 chain is already
   a fully sorted singly linked list, so this is just a walk. *)
let iter t (f : 'k -> 'v -> unit) : unit =
  let rec go = function
    | None -> ()
    | Some node -> f node.key node.value; go node.forward.(0)
  in
  go t.header.(0)

(* Same walk as [iter], but as a lazy [Seq.t] — closer to Rust's
   `Iterator`, and what merge-iterator work over several memtables/
   SSTables later will want to compose against. *)
let to_seq t : ('k * 'v) Seq.t =
  let rec go node () =
    match node with
    | None -> Seq.Nil
    | Some node -> Seq.Cons ((node.key, node.value), go node.forward.(0))
  in
  go t.header.(0)
```

A few design notes worth calling out:

- **The header is a plain forward-pointer array, not a node.** Textbook skip-list pseudocode usually gives the header a dummy/sentinel key so it can be treated uniformly with real nodes. In OCaml, that would mean manufacturing a fake `'k` value with no generic way to do so, so instead the header lives directly on `t` as `('k,'v) node option array`, and `find_predecessors` special-cases the very first hop before falling into uniform "walk a node's forward array" logic for everything after.
- **`insert` is upsert-only** — no new node, no level randomization, no size increment when the key already exists, matching the requirement that a memtable never holds two entries for the same key.
- **There's no `delete`, on purpose.** A deletion is represented as a live entry with an empty/tombstone value, not by removing anything from the skip list — that logic belongs one layer up, in the MemTable.
- **This is not thread-safe** — plain mutable arrays, no locks or atomics. Fine for a single mutable memtable in one domain; it's the piece that would need a genuine redesign (fine-grained per-node locking, not a couple of mutexes bolted on) to make concurrent, which is its own future post.

## Part 3: the MemTable, backed by the skip list

```ocaml
(* mem_table.ml *)

type t = {
  map : (bytes, bytes) Skiplist.t;
  wal : Wal.t option;
  id : int;
  mutable approximate_size : int;
}
```

This is a deliberately different shape from Part 1's `memtable_entry` — no separate `deleted` flag, no `timestamp` field, just raw `bytes -> bytes` with an *empty* `bytes` value standing in for a tombstone. That's not an inconsistency, it's what the actual `SkipMap<Bytes, Bytes>` in the Rust reference does at this stage of the tutorial: deletion is represented purely by writing an empty value, and per-entry timestamps don't show up until the later MVCC chapters. The earlier array-based version was a different reference point; this one is matching `mini-lsm` week 1 specifically.

A couple of things that *don't* need translating from the Rust struct at all: Rust wraps `map` and `approximate_size` in `Arc` so cloning a `MemTable` handle is cheap and every clone sees the same shared state. OCaml records are already heap-allocated, reference values — putting the same `t` in two places already means both point at the identical underlying skip list, no wrapping required. And `approximate_size` stays a plain `mutable int` rather than `AtomicUsize` for now: making just that one field atomic while the skip list underneath it isn't synchronized at all would be cosmetic, not real safety. Both of those get revisited together whenever the skip list becomes concurrent.

```ocaml
let create (id : int) : t =
  { map = Skiplist.create Bytes.compare; wal = None; id; approximate_size = 0 }

let get (t : t) (key : bytes) : bytes option =
  Skiplist.get t.map key
```

`Bytes.compare` gives the same byte-wise lexicographic ordering Rust's `Ord for Bytes` uses, so key order matches between the two implementations. `get` doesn't need any unwrapping — `Skiplist.get` already returns `'v option`, and with `'v = bytes` here that's already exactly `bytes option`, so it's a direct passthrough rather than a `match`.

`put` needed a bit more thought, because it's also supposed to flush to the write-ahead log, and the WAL as I'd originally written it (for a different project) appends newline-terminated *text* lines — which can't safely round-trip arbitrary key/value bytes that might themselves contain a `\n`. So `put` needs its own binary framing: `key_len (4 bytes, big-endian) | key | value_len (4 bytes, BE) | value`, with a zero-length value doubling as the tombstone marker, matching how the skip list already represents deletion. I added a raw, undelimited `Wal.append_bytes` alongside the existing text-based `append`, since the framing itself already marks record boundaries — no extra delimiter needed, and the original `append`/`log_drop` paths are untouched.

```ocaml
let put_uint32_be (buf : Buffer.t) (n : int) : unit =
  Buffer.add_char buf (Char.chr ((n lsr 24) land 0xff));
  Buffer.add_char buf (Char.chr ((n lsr 16) land 0xff));
  Buffer.add_char buf (Char.chr ((n lsr 8) land 0xff));
  Buffer.add_char buf (Char.chr (n land 0xff))

let encode_record (key : bytes) (value : bytes) : bytes =
  let buf = Buffer.create (8 + Bytes.length key + Bytes.length value) in
  put_uint32_be buf (Bytes.length key);
  Buffer.add_bytes buf key;
  put_uint32_be buf (Bytes.length value);
  Buffer.add_bytes buf value;
  Buffer.to_bytes buf

let put (t : t) (key : bytes) (value : bytes) : (unit, Wal.error) result =
  Skiplist.insert t.map key value;
  t.approximate_size <- t.approximate_size + Bytes.length key + Bytes.length value;
  match t.wal with
  | None -> Ok ()
  | Some wal -> Wal.append_bytes wal (encode_record key value)

let sync_wal (t : t) : (unit, Wal.error) result =
  match t.wal with
  | None -> Ok ()
  | Some wal -> Wal.sync wal
```

`approximate_size` here just adds unconditionally, even on an overwrite of an existing key — the array-backed version computed an exact delta by subtracting the old value's length first, but doing that here would mean a `Skiplist.get` before every `put`, roughly doubling the cost of every write. Given the field is explicitly named *approximate*, that felt like the right tradeoff, though it's a judgment call rather than something I've confirmed against the real Rust body (still `unimplemented!()` in the reference at this point).

The recovery side — replaying a WAL file back into a fresh skip list on startup — turned out to need the exact same binary framing on the read side, plus one thing worth remembering about any WAL: the *last* record in the file can legitimately be torn, half-written, if the process crashed mid-append. Recovery has to stop cleanly at that incomplete tail rather than raise an exception — that's the whole reason a WAL exists in the first place, so it needs to survive its own worst case gracefully.

## What's next

Still open, roughly in the order I'm planning to tackle them in the next few blogs:

- **Range scans** — `Skiplist.range` with a `Bound`-style lower/upper type, seeking to the start in O(log n) by reusing the same predecessor search `get`/`insert` already do, plus a small mutable cursor on top to match the iterator-style `next`/`key`/`value` shape the rest of the engine expects.

- **Flushing to an SSTable** — blocked on writing `SsTableBuilder` first (week 1, day 6 in the tutorial), so this stays a stub for now.

- **A real concurrent skip list** — the current one is explicitly single-domain and unsynchronized. The plan is a coarse-grained version first (the whole thing behind one OxCaml `Capsule`-guarded lock, for a statically race-free baseline), then, eventually, proper per-node lock coupling with optimistic traversal — the same shape as the algorithm behind Java's `ConcurrentSkipListMap` — once the coarse version is solid enough to build on.



# References 

- There were quite a few references I used, the [Rust Manual](https://skyzh.github.io/mini-lsm/00-overview.html) is what I will be following for the next series of blogs. I went through a few Skiplist implementation blogs such as the [OCaml one](https://ssojet.com/data-structures/implement-skip-list-in-ocaml#skip-list-structure-and-node-definition), [a great primer on Skiplists](https://rowjee.com/blog/skiplists) and [also this Rust implementation of it](https://danielorihuela.dev/blog/skip-list/). For the specific MemTable reference [I used this blog that implements it in Rust](https://adambcomer.com/blog/simple-database/memtable/). 

---
title:  Implementing a Mini LSM in OCaml- Part 2 
date: 2026-10-1
description: This blog implements a concurrent SkipList.
tags: [Data Structures, OCaml]
coverImage: /cover_images/cassette_tape.jpg
reference: https://en.wikipedia.org/wiki/Cassette_tape
referenceText: "cover image: a random cassette tape"
draft: false
---


# A Concurrent Skip List in OCaml, and a Bug I Haven't Fixed Yet

This is a follow-up to the Mini LSM series, but it's its own post. The single-threaded skip list from that series was always the "get it correct first" step, with a real concurrent, lock-based version as the actual target underneath the MemTable. This post is that attempt. I want to be upfront about how it went: I got it compiling, got it passing every sequential test I threw at it, then stress-tested it under real threads and found a genuine hang that I haven't fixed. I think that's more useful to write down honestly than to present a clean "here's my concurrent skip list" that might be quietly wrong.

## Where to start

Not at fine-grained locking. At one lock around the whole thing, first. The plan I settled on had three stages:

1. **A coarse-grained baseline**: wrap the entire single-threaded skip list behind one lock, so every operation serializes but is statically race-free. Zero parallelism, but a solid floor to build from.
2. **Fine-grained lock coupling**: per-node (or, practically, per-stripe) locks, unsynchronized lock-free reads, and an optimistic validate-then-mutate protocol for writers. This is the shape behind Java's `ConcurrentSkipListMap`, and closely modeled on Herlihy, Lev, Luchangco & Shavit's "A Simple Optimistic Skip-List Algorithm."
3. Eventually, revisit read-heavy workloads with something like a striped rwlock if profiling says the mutex version is still bottlenecked there.

This post is stage 2, attempted directly, with the coarse-grained version as a fallback I'm glad I have, precisely because stage 2 turned out to have a real bug in it.

## The design

Every node's mutable state moves to `Atomic.t`: its value, its forward pointers, a `marked` flag for logical deletion, and a `fully_linked` flag so a concurrent lock-free reader never observes a node that's been spliced in at one level but not the others yet.

```ocaml
type ('k, 'v) node = {
  key : 'k;
  value : 'v Atomic.t;
  top_level : int;
  next : ('k, 'v) node option Atomic.t array;
  marked : bool Atomic.t;
  fully_linked : bool Atomic.t;
  stripe : int;
}
```

`stripe` is the concession that makes this tractable. Real per-node locks (one mutex genuinely owned by each node) are what the reference algorithm uses, but OxCaml's `Capsule` API is built around a small number of statically-created lock brands, not one generated per node at runtime, so a literal per-node `Capsule.Mutex` doesn't map on cleanly. Instead, every node hashes deterministically to one of a fixed number of stripes (32 by default) at creation, and any operation touching more than one node's pointers locks the *distinct* stripes involved, always in ascending index order, to keep one consistent global lock ordering across every writer:

```ocaml
type ('k, 'v) t = {
  compare : 'k -> 'k -> int;
  max_level : int;
  p : float;
  num_stripes : int;
  stripes : Mutex.t array;
  level : int Atomic.t;
  header : ('k, 'v) node option Atomic.t array;
  size : int Atomic.t;
}
```

`get`, `contains_key`, `length`, `is_empty`, and `to_list` never take a lock at all. They walk purely through `Atomic.get`, the same shape as the single-threaded search, just reading atomically instead of through plain mutable fields. `insert` and `remove` are the only two operations that acquire anything: gather predecessors via an unsynchronized search, lock their distinct stripes in order, *validate* that nothing moved underneath since the search, then splice. Or, for `remove`, mark the node deleted first under its own stripe lock, then physically unlink it in a second, separately-locked pass. That two-phase "logical, then physical" deletion is the "lazy" half of the reference algorithm, and it's specifically what lets lock-free readers see a deletion immediately (via the `marked` flag) without ever needing to touch the locking machinery themselves.

```ocaml
(* skiplist_concurrent.mli *)
type ('k, 'v) t

val create : ?stripes:int -> ?max_level:int -> ?p:float -> ('k -> 'k -> int) -> ('k, 'v) t
val get : ('k, 'v) t -> 'k -> 'v option
val contains_key : ('k, 'v) t -> 'k -> bool
val insert : ('k, 'v) t -> 'k -> 'v -> unit
val remove : ('k, 'v) t -> 'k -> bool
val length : ('k, 'v) t -> int
val is_empty : ('k, 'v) t -> bool
val to_list : ('k, 'v) t -> ('k * 'v) list
```

```ocaml
(* skiplist_concurrent.ml *)

type ('k, 'v) node = {
  key : 'k;
  value : 'v Atomic.t;
  top_level : int;
  next : ('k, 'v) node option Atomic.t array;
  marked : bool Atomic.t;
  fully_linked : bool Atomic.t;
  stripe : int;
}

type ('k, 'v) t = {
  compare : 'k -> 'k -> int;
  max_level : int;
  p : float;
  num_stripes : int;
  stripes : Mutex.t array;
  level : int Atomic.t;
  header : ('k, 'v) node option Atomic.t array;
  size : int Atomic.t;
}

let create ?(stripes = 32) ?(max_level = 32) ?(p = 0.5) (compare : 'k -> 'k -> int) :
    ('k, 'v) t =
  {
    compare;
    max_level;
    p;
    num_stripes = stripes;
    stripes = Array.init stripes (fun _ -> Mutex.create ());
    level = Atomic.make 1;
    header = Array.init max_level (fun _ -> Atomic.make None);
    size = Atomic.make 0;
  }

let random_level t =
  let lvl = ref 1 in
  while Random.float 1.0 < t.p && !lvl < t.max_level do
    incr lvl
  done;
  !lvl

let stripe_of t key = Hashtbl.hash key mod t.num_stripes
let header_stripe = 0

let pred_stripe (pred : ('k, 'v) node option) : int =
  match pred with Some node -> node.stripe | None -> header_stripe

(* Unsynchronized traversal, reading only through [Atomic.get] - same shape
   as the single-threaded search. Marked nodes stay correctly positioned
   by key until physically unlinked, so the walk doesn't special-case
   them; only the terminal candidate check looks at [marked]/[fully_linked]. *)
let find_predecessors t key =
  let preds = Array.make t.max_level None in
  let succs = Array.make t.max_level None in
  let cur_arr = ref t.header in
  let cur_node = ref None in
  let level = Atomic.get t.level in
  for i = level - 1 downto 0 do
    let advancing = ref true in
    while !advancing do
      match Atomic.get !cur_arr.(i) with
      | Some next_node when t.compare next_node.key key < 0 ->
        cur_node := Some next_node;
        cur_arr := next_node.next
      | other ->
        succs.(i) <- other;
        advancing := false
    done;
    preds.(i) <- !cur_node
  done;
  (preds, succs)

let get t key : 'v option =
  let _, succs = find_predecessors t key in
  match succs.(0) with
  | Some node
    when t.compare node.key key = 0
         && Atomic.get node.fully_linked
         && not (Atomic.get node.marked) ->
    Some (Atomic.get node.value)
  | _ -> None

let contains_key t key = Option.is_some (get t key)

let lock_stripes t stripes_needed = List.iter (fun s -> Mutex.lock t.stripes.(s)) stripes_needed
let unlock_stripes t stripes_needed =
  List.iter (fun s -> Mutex.unlock t.stripes.(s)) stripes_needed

let rec insert t key value : unit =
  let preds, succs = find_predecessors t key in
  match succs.(0) with
  | Some node when t.compare node.key key = 0 ->
    Mutex.lock t.stripes.(node.stripe);
    if Atomic.get node.marked then begin
      Mutex.unlock t.stripes.(node.stripe);
      insert t key value
    end
    else begin
      Atomic.set node.value value;
      Mutex.unlock t.stripes.(node.stripe)
    end
  | _ ->
    let top_level = random_level t in
    let needed_stripes =
      List.init top_level (fun i -> pred_stripe preds.(i)) |> List.sort_uniq compare
    in
    lock_stripes t needed_stripes;
    let valid = ref true in
    for i = 0 to top_level - 1 do
      let actual =
        match preds.(i) with Some node -> Atomic.get node.next.(i) | None -> Atomic.get t.header.(i)
      in
      let pred_ok = match preds.(i) with Some node -> not (Atomic.get node.marked) | None -> true in
      if (not pred_ok) || actual != succs.(i) then valid := false
    done;
    if not !valid then begin
      unlock_stripes t needed_stripes;
      insert t key value
    end
    else begin
      if top_level > Atomic.get t.level then Atomic.set t.level top_level;
      let next = Array.init top_level (fun i -> Atomic.make succs.(i)) in
      let new_node =
        {
          key;
          value = Atomic.make value;
          top_level;
          next;
          marked = Atomic.make false;
          fully_linked = Atomic.make false;
          stripe = stripe_of t key;
        }
      in
      for i = 0 to top_level - 1 do
        match preds.(i) with
        | Some node -> Atomic.set node.next.(i) (Some new_node)
        | None -> Atomic.set t.header.(i) (Some new_node)
      done;
      Atomic.set new_node.fully_linked true;
      Atomic.incr t.size;
      unlock_stripes t needed_stripes
    end

let remove t key : bool =
  let _, succs = find_predecessors t key in
  match succs.(0) with
  | Some node when t.compare node.key key = 0 -> begin
      Mutex.lock t.stripes.(node.stripe);
      if Atomic.get node.marked || not (Atomic.get node.fully_linked) then begin
        Mutex.unlock t.stripes.(node.stripe);
        false
      end
      else begin
        Atomic.set node.marked true;
        Mutex.unlock t.stripes.(node.stripe);
        let rec unlink () =
          let preds', _ = find_predecessors t key in
          let needed_stripes =
            List.init node.top_level (fun i -> pred_stripe preds'.(i)) |> List.sort_uniq compare
          in
          lock_stripes t needed_stripes;
          let valid = ref true in
          for i = 0 to node.top_level - 1 do
            let points_at_node =
              match preds'.(i) with
              | Some p -> (match Atomic.get p.next.(i) with Some n -> n == node | None -> false)
              | None -> (match Atomic.get t.header.(i) with Some n -> n == node | None -> false)
            in
            if not points_at_node then valid := false
          done;
          if not !valid then begin
            unlock_stripes t needed_stripes;
            unlink ()
          end
          else begin
            for i = 0 to node.top_level - 1 do
              match preds'.(i) with
              | Some p -> Atomic.set p.next.(i) (Atomic.get node.next.(i))
              | None -> Atomic.set t.header.(i) (Atomic.get node.next.(i))
            done;
            unlock_stripes t needed_stripes
          end
        in
        unlink ();
        Atomic.decr t.size;
        true
      end
    end
  | _ -> false

let length t = Atomic.get t.size
let is_empty t = Atomic.get t.size = 0

let to_list t : ('k * 'v) list =
  let rec go acc node_opt =
    match node_opt with
    | None -> List.rev acc
    | Some node ->
      let acc =
        if Atomic.get node.fully_linked && not (Atomic.get node.marked) then
          (node.key, Atomic.get node.value) :: acc
        else acc
      in
      go acc (Atomic.get node.next.(0))
  in
  go [] (Atomic.get t.header.(0))
```

Two OCaml-specific traps worth naming, since I hit both while writing this. `Array.make n (Atomic.make None)` shares one single `Atomic.t` cell across all `n` slots. The initializer expression is evaluated once, not once per index, so every array of atomics here uses `Array.init` instead, which calls the initializer fresh per index. And physical-equality checks on `option`-wrapped nodes only mean what you want if both sides came from an actual `Atomic.get`, never from freshly constructing a literal `Some node` to compare against, since `Some x == Some x` is false in general (each `Some` is its own heap allocation). I made exactly that mistake once while writing `remove`'s validation and caught it re-reading the diff, not from the compiler. Nothing about `actual != Some node` looks wrong at a glance.

## Testing it, honestly

There's no OxCaml toolchain available to me in the environment I write these posts from, so what's below is checked against plain OCaml 4.14 with the `threads` library instead. This is enough to verify the code actually type-checks and to exercise real thread interleaving, but not a substitute for testing under genuine OCaml 5 multicore parallelism, let alone OxCaml's mode checker specifically.

**Sequential correctness: solid.** Running everything on a single thread (so the locking machinery still executes, just uncontested): insert a shuffled 2,000-key range, confirm every key round-trips, confirm `to_list` comes back strictly ascending, remove half by an ordering unrelated to key value, confirm what's left is exactly right and still sorted. Passed clean across every random seed I tried.

**Concurrent stress testing found a real hang.** Eight threads, 20,000 mixed insert/remove/get operations each, contending over a pool of only 500 keys, with deliberately high contention. It hung reliably, every time, well before any thread finished.

I instrumented lock acquisition (thread id, which stripes, logged before and after `Mutex.lock`) to tell whether this was a classic deadlock (two threads each waiting on a lock the other holds) or a livelock, spinning forever without ever actually blocking. Every *instrumented* lock attempt I logged was eventually matched by a successful acquisition; nothing was left permanently waiting on the multi-stripe locking path specifically, which argues against a lock-ordering deadlock there. But the thread that hung produced zero instrumented output at all, meaning it likely never reached that path in the first place; it's stuck somewhere in one of the single-lock fast paths (updating an already-existing key in `insert`, or the initial mark step in `remove`), neither of which I'd instrumented. I tried adding `Thread.yield ()` before each retry, on the theory it was pure contention-driven livelock; it visibly helped some threads make more progress before the run still eventually hung, which tells me backoff alone isn't the fix, something about the protocol itself is off, not just the scheduling.

My working hypothesis, without full confidence: an `insert` retrying against a `remove`-marked node that's slow to actually finish being physically unlinked, so the insert can never proceed past "this key exists but is marked" to either updating it or inserting fresh. I was not able to pin this to a fully confident root cause through the reasoning and testing available to me in this environment, and I'd rather say that plainly than hand over a diagnosis I can't fully back.

## Where this leaves things

The types, the lock-free read path, and the overall stripe-locking structure I'm fairly confident in; every sequential test exercises them correctly. `insert`/`remove` under real contention are where I'd put a firm "known issue, don't trust this with concurrent writers yet" sign. I'm picking this back up with `memtrace` and possibly `eio`'s tooling to get a proper look at where it's actually stuck, rather than continuing to reason about it print-statement by print-statement.
## A benchmark, and an independent confirmation of the bug

After the last post, two things happened: I built a benchmark comparing this against the original single-threaded `Skiplist`, and separately kept trying to reproduce the hang. The benchmark numbers are genuinely encouraging. The reproduction attempt turned into the more important result.

### The benchmark

One thing worth getting right before trusting any of these numbers: real parallelism in OCaml 5 comes from `Domain`, not `Thread`. `Thread` still runs under one cooperative runtime lock. Spawning more of them never gets you actual multi-core execution, no matter how good the locking design underneath is. So the scaling half of this benchmark uses `Domain.spawn`, not `Thread.create`, specifically so a bad number means something real about the data structure rather than just "I benchmarked the wrong primitive."

**Single-threaded, no contention possible: `Skiplist` vs `Concurrent_skip`, 100,000 keys:**

```
Skiplist: insert                                 0.054s
Skiplist: get (every key)                        0.044s
Concurrent_skip: insert (1 domain)               0.094s
Concurrent_skip: get (1 domain, every key)        0.087s
```

`Concurrent_skip` runs roughly 1.5-1.8x slower than `Skiplist` here, uncontended, and that's expected rather than concerning. Every field that's `Atomic.t` (the value, each forward pointer, the marked/fully-linked flags) is its own separately heap-allocated cell, where `Skiplist`'s plain `mutable` fields sit inline in the node record with no indirection at all. `insert` also computes a stripe hash and takes a `Mutex.lock`/`unlock` even with nobody else around to contend with. Modern mutexes have a cheap uncontended fast path, but "cheap" isn't "free". This is the honest, unavoidable cost of buying thread-safety, paid whether or not you ever run with more than one domain.

**Scaling: `Concurrent_skip` only, 50,000 ops per domain, key space of 50,000:**

```
 1 domains:    0.022s total,    2337988 ops/sec
 2 domains:    0.028s total,    3617307 ops/sec
 4 domains:    0.035s total,    5828377 ops/sec
 8 domains:    0.072s total,    5518767 ops/sec
```

1→2 domains gets ~1.5x throughput, 1→4 gets ~2.4x. Sublinear, but that's the expected shape for a striped-lock design, not a red flag on its own: 32 stripes over a small key space means some domains collide on the same stripe by chance, and pushing more domains at a *fixed* key space mechanically raises contention as domain count goes up, independent of anything about the implementation. On its own, this would be a genuinely solid result.

### It isn't on its own, though

Run that same benchmark repeatedly, and the 8-domain line doesn't always show up. Sometimes it completes in the ~0.07s you'd expect from the trend. Sometimes it doesn't finish inside a 1000-second timeout at all.

That gap is the important part, not the throughput number. There's no ordinary slowdown that turns "finishes in 0.07s" into "doesn't finish in over 1000 seconds". A genuine 10x or even 1000x contention penalty would still land well under a minute. A bimodal split like that (either fast, or apparently stuck) is the signature of a probabilistic scheduling bug: most interleavings resolve cleanly, and some fraction of the time, the scheduler happens to produce the exact racy ordering that gets something stuck.

This is worth sitting with rather than waving off, and specifically *because* it's intermittent rather than in spite of it. A bug that fails every single time gets caught before it ships. A bug that fails one time in four passes a test suite most of the time, ships, and then locks up a running process on some unremarkable day, with nothing having changed. Intermittent is the property that makes a concurrency bug genuinely dangerous, not a mitigating detail.

What this run does give me, though, is something more valuable than another data point from my own sandboxed testing: an independent reproduction, on real hardware, under genuine OCaml 5 `Domain`-based parallelism, of the exact same failure shape I found earlier under OCaml 4.14's cooperative threads. It was fine at low contention, breaks specifically once domain count and contention both climb. Two different environments landing on the same trigger conditions is real corroborating evidence this lives in the locking protocol itself, not in some quirk of either machine.

Next concrete step: catching it live with `sample`/`spindump` (or the equivalent) the moment it's stuck, to get an actual native stack trace per domain rather than continuing to infer blocked-vs-spinning indirectly. That's a more direct answer than anything memtrace or repeated benchmark runs can give on their own. Still open, but now with a real, reproducible way to go looking.

## Trying dscheck: exhaustive interleaving search instead of hoping to get lucky

Stress testing can only ever demonstrate a bug's presence, never its absence. A clean run is a data point, not proof. What I actually want is something that can say "no bug exists in this scenario," and that's a model checker's job, not a stress harness's. [`dscheck`](https://github.com/ocaml-multicore/dscheck) does exactly this for concurrent OCaml: instead of running your program once with whatever interleaving the scheduler happens to produce, it exhaustively explores *every* possible interleaving of a small scenario and checks your assertions against all of them. It's the same tool Jane Street/Tarides use to verify their own `Saturn` concurrent-collections library, about as close to "built for this exact situation" as it gets.

There's a real constraint buried in its docs that changes the shape of the work, though:

> Domains can communicate through atomic variables only... Tested programs have to be at least lock-free. If any thread cannot finish on its own, DSCheck will explore its transitions ad infinitum.

A blocked thread waiting on a genuine OS `Mutex` isn't a state transition dscheck can see at all. It'll just spin forever trying to explore "still waiting," never reaching a terminal state to check. So "point dscheck at `Concurrent_skip` as it stands" was never going to work. What has to happen instead is rebuilding the same protocol so it communicates exclusively through atomics, including the locking itself, without ending up testing a different program than the one that actually ships.

### Making the protocol swappable, not rewriting it

The fix: replace the stripe `Mutex.t` with a CAS-based spinlock built purely from `compare_and_set`, and parameterize the whole module over both its atomic operations and its lock, so the exact same insert/remove/get logic runs against real `Stdlib.Atomic` + real `Mutex` in production, and against `Dscheck.TracedAtomic` + a traced spinlock under the checker.

```ocaml
(* concurrent_skip_intf.ml *)

module type ATOMIC = sig
  type 'a t
  val make : 'a -> 'a t
  val get : 'a t -> 'a
  val set : 'a t -> 'a -> unit
  val compare_and_set : 'a t -> 'a -> 'a -> bool
end

module type LOCK = sig
  type t
  val create : unit -> t
  val lock : t -> unit
  val unlock : t -> unit
end

(* Production: real OS mutex - blocks, yields the core, doesn't burn CPU
   while waiting. *)
module Mutex_lock : LOCK = struct
  type t = Mutex.t
  let create = Mutex.create
  let lock = Mutex.lock
  let unlock = Mutex.unlock
end

(* Test-only: a spinlock built from nothing but compare_and_set on the
   SAME atomic module the rest of the structure uses, so under dscheck,
   lock acquisition is just more traced, explorable atomic operations -
   never use this in production, busy-spinning is a genuinely worse
   choice than blocking whenever you're not specifically doing this for
   model-checking. *)
module Make_spinlock (A : ATOMIC) : LOCK with type t = bool A.t = struct
  type t = bool A.t
  let create () = A.make false
  let lock t =
    let acquired = ref false in
    while not !acquired do
      acquired := A.compare_and_set t false true
    done
  let unlock t = A.set t false
end
```

```ocaml
(* concurrent_skip.ml - same algorithm as before, now parameterized over
   how it talks to shared memory rather than hardcoded to Stdlib.Atomic
   and Mutex *)

module Make (A : ATOMIC) (L : LOCK) = struct
  type ('k, 'v) node = {
    key : 'k;
    value : 'v A.t;
    top_level : int;
    next : ('k, 'v) node option A.t array;
    marked : bool A.t;
    fully_linked : bool A.t;
    stripe : int;
  }

  type ('k, 'v) t = {
    compare : 'k -> 'k -> int;
    max_level : int;
    p : float;
    num_stripes : int;
    stripes : L.t array;
    level : int A.t;
    header : ('k, 'v) node option A.t array;
    size : int A.t;
  }

  let create ?(stripes = 32) ?(max_level = 32) ?(p = 0.5) compare =
    {
      compare; max_level; p; num_stripes = stripes;
      stripes = Array.init stripes (fun _ -> L.create ());
      level = A.make 1;
      header = Array.init max_level (fun _ -> A.make None);
      size = A.make 0;
    }

  let random_level t =
    let lvl = ref 1 in
    while Random.float 1.0 < t.p && !lvl < t.max_level do incr lvl done;
    !lvl

  let stripe_of t key = Hashtbl.hash key mod t.num_stripes
  let header_stripe = 0
  let pred_stripe pred = match pred with Some node -> node.stripe | None -> header_stripe

  let find_predecessors t key =
    let preds = Array.make t.max_level None in
    let succs = Array.make t.max_level None in
    let cur_arr = ref t.header in
    let cur_node = ref None in
    let level = A.get t.level in
    for i = level - 1 downto 0 do
      let advancing = ref true in
      while !advancing do
        match A.get !cur_arr.(i) with
        | Some next_node when t.compare next_node.key key < 0 ->
          cur_node := Some next_node;
          cur_arr := next_node.next
        | other -> succs.(i) <- other; advancing := false
      done;
      preds.(i) <- !cur_node
    done;
    (preds, succs)

  let get t key =
    let _, succs = find_predecessors t key in
    match succs.(0) with
    | Some node when t.compare node.key key = 0 && A.get node.fully_linked && not (A.get node.marked) ->
      Some (A.get node.value)
    | _ -> None

  let contains_key t key = Option.is_some (get t key)

  let lock_stripes t stripes_needed = List.iter (fun s -> L.lock t.stripes.(s)) stripes_needed
  let unlock_stripes t stripes_needed = List.iter (fun s -> L.unlock t.stripes.(s)) stripes_needed

  let rec insert t key value =
    let preds, succs = find_predecessors t key in
    match succs.(0) with
    | Some node when t.compare node.key key = 0 ->
      L.lock t.stripes.(node.stripe);
      if A.get node.marked then begin
        L.unlock t.stripes.(node.stripe);
        insert t key value
      end else begin
        A.set node.value value;
        L.unlock t.stripes.(node.stripe)
      end
    | _ ->
      let top_level = random_level t in
      let needed_stripes = List.init top_level (fun i -> pred_stripe preds.(i)) |> List.sort_uniq compare in
      lock_stripes t needed_stripes;
      let valid = ref true in
      for i = 0 to top_level - 1 do
        let actual = match preds.(i) with Some node -> A.get node.next.(i) | None -> A.get t.header.(i) in
        let pred_ok = match preds.(i) with Some node -> not (A.get node.marked) | None -> true in
        if (not pred_ok) || actual != succs.(i) then valid := false
      done;
      if not !valid then begin
        unlock_stripes t needed_stripes;
        insert t key value
      end else begin
        if top_level > A.get t.level then A.set t.level top_level;
        let next = Array.init top_level (fun i -> A.make succs.(i)) in
        let new_node = {
          key; value = A.make value; top_level; next;
          marked = A.make false; fully_linked = A.make false; stripe = stripe_of t key;
        } in
        for i = 0 to top_level - 1 do
          match preds.(i) with
          | Some node -> A.set node.next.(i) (Some new_node)
          | None -> A.set t.header.(i) (Some new_node)
        done;
        A.set new_node.fully_linked true;
        A.set t.size (A.get t.size + 1);
        unlock_stripes t needed_stripes
      end

  let remove t key =
    let _, succs = find_predecessors t key in
    match succs.(0) with
    | Some node when t.compare node.key key = 0 -> begin
        L.lock t.stripes.(node.stripe);
        if A.get node.marked || not (A.get node.fully_linked) then begin
          L.unlock t.stripes.(node.stripe);
          false
        end else begin
          A.set node.marked true;
          L.unlock t.stripes.(node.stripe);
          let rec unlink () =
            let preds', _ = find_predecessors t key in
            let needed_stripes = List.init node.top_level (fun i -> pred_stripe preds'.(i)) |> List.sort_uniq compare in
            lock_stripes t needed_stripes;
            let valid = ref true in
            for i = 0 to node.top_level - 1 do
              let points_at_node =
                match preds'.(i) with
                | Some p -> (match A.get p.next.(i) with Some n -> n == node | None -> false)
                | None -> (match A.get t.header.(i) with Some n -> n == node | None -> false)
              in
              if not points_at_node then valid := false
            done;
            if not !valid then begin unlock_stripes t needed_stripes; unlink () end
            else begin
              for i = 0 to node.top_level - 1 do
                match preds'.(i) with
                | Some p -> A.set p.next.(i) (A.get node.next.(i))
                | None -> A.set t.header.(i) (A.get node.next.(i))
              done;
              unlock_stripes t needed_stripes
            end
          in
          unlink ();
          A.set t.size (A.get t.size - 1);
          true
        end
      end
    | _ -> false

  let length t = A.get t.size
  let is_empty t = A.get t.size = 0

  let to_list t =
    let rec go acc node_opt =
      match node_opt with
      | None -> List.rev acc
      | Some node ->
        let acc = if A.get node.fully_linked && not (A.get node.marked) then (node.key, A.get node.value) :: acc else acc in
        go acc (A.get node.next.(0))
    in
    go [] (A.get t.header.(0))
end

(* the production module everything else keeps using, unchanged from
   the outside *)
module Concurrent_skip = Make (Stdlib.Atomic) (Mutex_lock)
```

Only real behavioral change from the earlier version: `Atomic.incr`/`Atomic.decr` on `t.size` became explicit `get`-then-`set`, since `Dscheck.TracedAtomic` doesn't necessarily expose `incr`/`decr` as their own traced primitives - spelling it out keeps both instantiations honestly doing the same thing at the atomic level rather than assuming the convenience functions exist on both sides.

### A deliberately tiny, targeted test

The other calibration worth being explicit about: dscheck scales to a handful of domains and operations, full stop. Exhaustive search means the state space is the whole cost. Don't try to check "8 domains, 20,000 ops," the scenario the stress test hammers; go straight at the smallest scenario matching the actual suspected bug: a `remove` racing an `insert` on the *same* key:

```ocaml
(* test/dscheck_concurrent_skip.ml *)

module Cs = Concurrent_skip.Make (Dscheck.TracedAtomic) (Concurrent_skip.Make_spinlock (Dscheck.TracedAtomic))

let test_insert_remove_same_key_race () =
  let t = Cs.create compare in
  Dscheck.TracedAtomic.spawn (fun () -> Cs.insert t 1 "a"; ignore (Cs.remove t 1));
  Dscheck.TracedAtomic.spawn (fun () -> Cs.insert t 1 "b");
  Dscheck.TracedAtomic.final (fun () ->
      Dscheck.TracedAtomic.check (fun () ->
          let entries = Cs.to_list t in
          let keys = List.map fst entries in
          List.length keys = List.length (List.sort_uniq compare keys)
          && Cs.length t = List.length entries
          && List.for_all (fun (k, v) -> Cs.get t k = Some v) entries))

let () = Dscheck.TracedAtomic.trace test_insert_remove_same_key_race
```

### Where this actually stands right now

I haven't gotten a clean result out of this yet. Running it currently produces no visible output at all, which is genuinely ambiguous rather than a bad sign on its own: a dscheck run that explores everything and finds zero violations may not print anything by design (its own examples only show output on a *found* violation), so "silent" could mean "clean pass on this narrow scenario" just as easily as "crashed before printing anything" or "still exploring." I'm adding explicit flushed prints around the `trace` call to tell those apart before drawing any conclusion from it. Whatever it turns out to be, it won't be the final word either way. A clean pass here would only rule out this one specific two-domain, single-key race, not the bug in general.

### References 

Not much to refer except for the [part 1 of the blog](https://sycamoreq.github.io/notes/implementing_mini_lsm/). I looked into dscheck as well as [meio](https://github.com/ocaml-multicore/meio) which is another CLI tool that monitors programs. 
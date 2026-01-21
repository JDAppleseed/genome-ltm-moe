# Phase dependency graph

```
Phase 0
  |
Phase 1
  |
Phase 2
  |
Phase 3
  |
Phase 4 ---- Phase 5 ---- Phase 6
   \           |           /
    \          |          /
      ---- Phase 7 ---- Phase 8 ---- Phase 9
                    |
                 Phase 10
                    |
                 Phase 11
                    |
                 Phase 12
                    |
                 Phase 13
                    |
                 Phase 14
```

Phases 0–3 are prerequisites. Phases 4–6 run in parallel after 0–3. Phases 7–9 depend on 4–6.
Phases 10–11 depend on 0–9, followed by final integration in phases 12–14.

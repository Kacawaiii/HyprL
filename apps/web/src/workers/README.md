# Web workers

Reserved for expensive **presentation** transforms — secondary downsampling of
an already-bounded series, parsing a large response, deriving a chart view.

Nothing that decides anything goes here. Signal direction, strength, position
sizing and every metric are computed in Python and arrive as data; moving any of
that into a worker would create a second implementation that can drift from the
engines, and it would drift silently because a worker fails quietly.

Empty by design: the server already caps chart responses at
`MAX_CHART_POINTS`, so there is currently no transform heavy enough to justify
the transfer cost.

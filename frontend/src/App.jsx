import { useMemo, useState } from 'react'
import logoUrl from './assets/logo.svg'
import { PLAYERS } from './data/players.js'

const ROLES = ['ALL', 'FW', 'MF', 'DF', 'GK']
const ROLE_LABELS = { FW: 'Forwards', MF: 'Midfield', DF: 'Defence', GK: 'Keepers' }
const ROLE_COLORS = { FW: 'var(--green)', MF: 'var(--lilac)', DF: 'var(--blue)', GK: 'var(--amber)' }

function initials(name) {
  const parts = name.split(' ').filter(Boolean)
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
}

function PixelDeco({ role }) {
  // deterministic scatter per role so it doesn't jump on re-render
  const seed = { FW: [0, 4, 5, 8], MF: [1, 3, 6, 7], DF: [2, 4, 7, 8], GK: [0, 2, 5, 6] }[role] || [0, 4, 8]
  return (
    <div className="podium-pixels" aria-hidden="true">
      {Array.from({ length: 9 }).map((_, i) => (
        <span
          key={i}
          style={seed.includes(i) ? { background: ROLE_COLORS[role], borderColor: ROLE_COLORS[role] } : undefined}
        />
      ))}
    </div>
  )
}

function Podium({ players }) {
  return (
    <div className="podium">
      {players.map((p) => (
        <div className="podium-card" key={p.name}>
          <PixelDeco role={p.role} />
          <div className="rank-num mono">RANK — {String(p.rank).padStart(2, '0')}</div>
          <div className="big-monogram" style={{ background: ROLE_COLORS[p.role] }}>
            {initials(p.name)}
          </div>
          <h3>{p.name}</h3>
          <div className="nation">
            {p.nation}
            {p.club ? <span className="club-badge">{p.club}</span> : null}
          </div>
          <div className="chips">
            <span className={`chip role-${p.role}`}>{p.role}</span>
            <span className="chip rating">{p.rating.toFixed(1)}</span>
          </div>
          <div className="rating-bar">
            <div className={`fill role-${p.role}`} style={{ width: `${(p.rating / 5) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function PlayerCard({ p }) {
  return (
    <div className="player-card">
      <div className="top-row">
        <div className={`monogram mg-${p.role}`}>{initials(p.name)}</div>
        <span className="rank-tag mono">#{String(p.rank).padStart(2, '0')}</span>
      </div>
      <h4>{p.name}</h4>
      <div className="nation">
        {p.nation}
        {p.club ? <span className="club-badge">{p.club}</span> : null}
      </div>
      <div className="meta-row">
        <div className="chips">
          <span className={`chip role-${p.role}`}>{p.role}</span>
          <span className="chip rating" title="Percentile within position, mapped to 1.0–5.0">
            {p.rating.toFixed(1)}
          </span>
        </div>
        <span className="score" title="Rank within position">
          #{p.rankInRole} in {p.role}
        </span>
      </div>
      <div className="card-foot mono">
        <span>{p.age}y · {p.minutes}′</span>
        <span>{p.gkAvailabilityOnly ? 'availability only' : `xGI/90 ${p.xgi90.toFixed(2)}`}</span>
      </div>
      <div className="rating-bar">
        <div className={`fill role-${p.role}`} style={{ width: `${(p.rating / 5) * 100}%` }} />
      </div>
    </div>
  )
}

function Distribution() {
  const buckets = useMemo(() => {
    const edges = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.01]
    const byRole = {}
    for (const role of ['FW', 'MF', 'DF', 'GK']) {
      const players = PLAYERS.filter((p) => p.role === role)
      const counts = edges.slice(0, -1).map((lo, i) => players.filter((p) => p.rating >= lo && p.rating < edges[i + 1]).length)
      const avg = players.reduce((s, p) => s + p.rating, 0) / (players.length || 1)
      byRole[role] = { counts, n: players.length, avg, max: Math.max(...counts, 1) }
    }
    return byRole
  }, [])

  return (
    <section className="dist" id="model">
      <div className="section-head">
        <h2>Rating distribution by role</h2>
        <span className="count mono">PERCENTILE RANK WITHIN ROLE → 1.0–5.0</span>
      </div>
      <div className="dist-grid">
        {['FW', 'MF', 'DF', 'GK'].map((role) => {
          const d = buckets[role]
          return (
            <div className="dist-cell" key={role}>
              <div className="role-name mono" style={{ color: ROLE_COLORS[role] }}>
                {ROLE_LABELS[role].toUpperCase()}
              </div>
              <div className="dist-bars">
                {d.counts.map((c, i) => (
                  <div
                    key={i}
                    className="bar"
                    style={{ height: `${(c / d.max) * 100}%`, background: ROLE_COLORS[role] }}
                    title={`${c} players`}
                  />
                ))}
              </div>
              <div className="dist-foot">
                <span>n = {d.n}</span>
                <span>avg {d.avg.toFixed(2)}</span>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default function App() {
  const [role, setRole] = useState('ALL')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('rating')
  const [plOnly, setPlOnly] = useState(false)

  const nations = useMemo(() => new Set(PLAYERS.map((p) => p.nation)).size, [])
  const plCount = useMemo(() => PLAYERS.filter((p) => p.inPL).length, [])

  const filtered = useMemo(() => {
    let list = PLAYERS
    if (plOnly) list = list.filter((p) => p.inPL)
    if (role !== 'ALL') list = list.filter((p) => p.role === role)
    if (query.trim()) {
      const q = query.trim().toLowerCase()
      list = list.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.nation.toLowerCase().includes(q) ||
          (p.club || '').toLowerCase().includes(q),
      )
    }
    const sorted = [...list]
    if (sort === 'rating') sorted.sort((a, b) => b.score - a.score)
    if (sort === 'name') sorted.sort((a, b) => a.name.localeCompare(b.name))
    if (sort === 'rating-asc') sorted.sort((a, b) => a.score - b.score)
    return sorted
  }, [role, query, sort, plOnly])

  const top3 = filtered.slice(0, 3)
  const showPodium = role === 'ALL' && !query.trim() && sort === 'rating'

  return (
    <>
      <div className="grid-bg" aria-hidden="true" />
      <div className="app">
        <div className="shell">
          <nav className="nav">
            <div className="nav-brand">
              <img src={logoUrl} alt="" className="brand-mark" aria-hidden="true" />
              XGAUGES
            </div>
            <div className="nav-links">
              <a href="#players">Players</a>
              <a href="#model">Model</a>
              <a href="https://github.com/DerrickWawerumuturi/epl-predictor" target="_blank" rel="noreferrer">
                GitHub ↗
              </a>
            </div>
          </nav>

          <header className="hero">
            <div className="pixels" aria-hidden="true">
              {['f4','','f1','','','f2','','f4','f3','','f4','','','f1','','f2'].map((c, i) => (
                <span key={i} className={c} />
              ))}
            </div>
            <div className="eyebrow mono">
              Euro 2024 · impact index
              <span className="model-chip">
                <span className="k">IDX</span>
                <span className="v">v2</span>
              </span>
            </div>
            <h1>
              Decoding player performance <span className="accent">before kickoff.</span>
            </h1>
            <p className="hero-sub">
              {PLAYERS.length} Euro 2024 players scored on expected goal involvement, output and
              minutes — then ranked <strong>against their own position</strong>. All leagues;{' '}
              {plCount} of them play in the Premier League.
            </p>
            <div className="hero-ctas">
              <a className="btn primary" href="#players">
                Explore the rankings →
              </a>
              <a className="btn" href="https://github.com/DerrickWawerumuturi/epl-predictor" target="_blank" rel="noreferrer">
                Read the code ↗
              </a>
            </div>
          </header>

          <div className="stats">
            <div className="stat">
              <div className="label">Players ranked</div>
              <div className="value">{PLAYERS.length}</div>
            </div>
            <div className="stat">
              <div className="label">Nations</div>
              <div className="value">{nations}</div>
            </div>
            <div className="stat">
              <div className="label">In the Premier League</div>
              <div className="value">
                <em>{plCount}</em> of {PLAYERS.length}
              </div>
            </div>
            <div className="stat">
              <div className="label">Rating</div>
              <div className="value">
                percentile <em>in role</em>
              </div>
            </div>
          </div>

          <div className="method-note mono">
            <strong>What this is:</strong> a transparent weighted index — 0.45 expected goal
            involvement per 90, 0.20 actual goals + assists per 90, 0.35 share of team minutes,
            each z-scored within position, minimum 270 Euro minutes. It <em>describes</em> Euro
            2024; it is not a prediction. Goalkeepers are scored on availability only.
            <br />
            <br />
            <strong>Why players from Real Madrid and Al-Nassr are here:</strong> the index needs no
            Premier League data, so it rates every Euro 2024 player regardless of league — only{' '}
            {plCount} of {PLAYERS.length} play in England. Use the{' '}
            <em>Premier League only</em> filter to narrow it. The separate supervised forecast is
            England-only, which is exactly why its sample is small.
          </div>

          <section id="players">
            <div className="section-head">
              <h2>Player rankings</h2>
              <span className="count mono">
                {filtered.length} / {PLAYERS.length} PLAYERS
              </span>
            </div>

            <div className="controls">
              <div className="role-tabs">
                {ROLES.map((r) => (
                  <button
                    key={r}
                    className={`role-tab ${role === r ? `active ${r === 'ALL' ? 'all' : r}` : ''}`}
                    onClick={() => setRole(r)}
                  >
                    {r}
                  </button>
                ))}
              </div>
              <input
                className="search"
                placeholder="Search player or nation…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <select className="sort-select" value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="rating">Sort: best first</option>
                <option value="rating-asc">Sort: worst first</option>
                <option value="name">Sort: name A–Z</option>
              </select>
              <button
                className={`pl-toggle mono ${plOnly ? 'on' : ''}`}
                onClick={() => setPlOnly((v) => !v)}
                aria-pressed={plOnly}
              >
                <span className="tick">{plOnly ? '✓' : ''}</span>
                Premier League only ({plCount})
              </button>
            </div>

            {showPodium && <Podium players={top3} />}

            <div className="player-grid">
              {filtered.length === 0 && <div className="empty">NO PLAYERS MATCH — TRY ANOTHER SEARCH</div>}
              {filtered.map((p) => (
                <PlayerCard key={p.name} p={p} />
              ))}
            </div>
          </section>

          <Distribution />

          <footer className="footer">
            <span>XGAUGES · built by Derrick Waweru</span>
            <span>
              data: FBref · Euro 2024 →{' '}
              <a href="https://github.com/DerrickWawerumuturi/epl-predictor" target="_blank" rel="noreferrer">
                epl-predictor
              </a>
            </span>
          </footer>
        </div>
      </div>
    </>
  )
}

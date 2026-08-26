/* ==========================================================================
   hud.js — the top bar: treasury counter, villager roster, chronicle rail,
   listing cards and toasts. Exposes window.Hud.
   ========================================================================== */
(function () {
  'use strict';

  const esc = window.escapeHtml;

  const el = {
    roster: document.getElementById('roster'),
    amount: document.getElementById('treasury-amount'),
    month: document.getElementById('treasury-month'),
    sparks: document.getElementById('treasury-sparks'),
    treasuryToggle: document.getElementById('treasury-toggle'),
    ledger: document.getElementById('ledger'),
    ledgerRows: document.getElementById('ledger-rows'),
    ledgerNote: document.getElementById('ledger-note'),
    chipMode: document.getElementById('chip-mode'),
    chipPublished: document.getElementById('chip-published'),
    chipShorts: document.getElementById('chip-shorts'),
    chipPending: document.getElementById('chip-pending'),
    chipLink: document.getElementById('chip-link'),
    log: document.getElementById('log'),
    listings: document.getElementById('listings'),
    listingsCount: document.getElementById('listings-count'),
    shorts: document.getElementById('shorts'),
    cinemaStats: document.getElementById('cinema-stats'),
    cinemaCount: document.getElementById('tab-cinema-count'),
    toasts: document.getElementById('toasts'),
    clearLog: document.getElementById('btn-clear-log'),
    newShort: document.getElementById('btn-new-short'),
    newDeal: document.getElementById('btn-new-deal'),
    deals: document.getElementById('deals'),
    dealsStats: document.getElementById('deals-stats'),
    dealsCount: document.getElementById('tab-deals-count'),
    tabs: Array.from(document.querySelectorAll('.tab')),
    panels: {
      chronicle: document.getElementById('panel-chronicle'),
      cinema: document.getElementById('panel-cinema'),
      deals: document.getElementById('panel-deals'),
    },
  };

  const MAX_LOG = 120;

  const Hud = {
    villagers: [],
    /** id -> badge element, so updates never re-query the DOM. */
    badges: new Map(),
    displayedRevenue: 0,
    /** The most recent stats payload, so panels can read totals lazily. */
    stats: null,
    shorts: [],
    onSelectVillager: null,
    onListingAction: null,
    onShortAction: null,
    onGenerateShort: null,
    /** Set by village.js: approve/reject a deal, record its metrics, scout one. */
    onDealAction: null,
    onDealMetrics: null,
    onScoutDeal: null,
    /** The most recent deals payload. */
    deals: [],

    /* ---------------------------------------------------------- roster */

    renderRoster(villagers, drawAvatar) {
      this.villagers = villagers;
      el.roster.innerHTML = '';
      this.badges.clear();

      villagers.forEach((villager) => {
        const item = document.createElement('li');
        item.className = 'villager-badge';
        item.dataset.agent = villager.id;
        item.title = `${villager.name} — ${villager.title}\nClick to centre the camera`;

        const avatar = document.createElement('div');
        avatar.className = 'villager-badge__avatar';
        avatar.style.background = villager.color;
        // A tiny canvas portrait if the engine supplies one; emoji otherwise.
        if (typeof drawAvatar === 'function') {
          const canvas = document.createElement('canvas');
          canvas.width = 26;
          canvas.height = 26;
          canvas.style.width = '26px';
          canvas.style.height = '26px';
          drawAvatar(canvas.getContext('2d'), villager.id, 26);
          avatar.appendChild(canvas);
        } else {
          avatar.textContent = villager.emoji;
        }

        const body = document.createElement('div');
        body.className = 'villager-badge__body';
        body.innerHTML =
          `<span class="villager-badge__name">${esc(villager.name)}</span>` +
          `<span class="villager-badge__task" data-role="task">${esc(villager.title)}</span>`;

        const dot = document.createElement('span');
        dot.className = 'villager-badge__dot status-idle';
        dot.dataset.role = 'dot';

        const progress = document.createElement('span');
        progress.className = 'villager-badge__progress';
        progress.dataset.role = 'progress';

        item.append(avatar, body, dot, progress);
        item.addEventListener('click', () => {
          window.Sfx.select();
          if (this.onSelectVillager) this.onSelectVillager(villager.id);
        });

        el.roster.appendChild(item);
        this.badges.set(villager.id, item);
      });
    },

    /** Reflect one agent's live state on its badge. */
    updateAgent(agent) {
      const badge = this.badges.get(agent.id);
      if (!badge) return;

      const dot = badge.querySelector('[data-role="dot"]');
      const task = badge.querySelector('[data-role="task"]');
      const progress = badge.querySelector('[data-role="progress"]');

      dot.className = 'villager-badge__dot status-' + agent.status;
      task.textContent = agent.task || '';
      task.title = agent.task || '';
      progress.style.width =
        agent.status === 'working' ? Math.round((agent.progress || 0.15) * 100) + '%' : '0%';

      badge.classList.toggle('is-active', agent.status === 'working');
    },

    markActive(agentId) {
      this.badges.forEach((badge, id) => {
        badge.classList.toggle('is-focused', id === agentId);
      });
    },

    /* -------------------------------------------------------- treasury */

    updateStats(stats) {
      if (!stats) return;
      this.stats = stats;

      el.month.textContent = stats.month || 'this month';
      el.chipPublished.textContent = '📦 ' + (stats.publishedTotal || 0);
      el.chipPending.textContent = '⏳ ' + (stats.pending || 0);
      el.chipShorts.textContent = '🎬 ' + ((stats.shorts && stats.shorts.published) || 0);
      el.chipShorts.title = stats.shorts
        ? `${stats.shorts.total || 0} Short(s), ${stats.shorts.published || 0} published`
        : 'Shorts published';

      // The treasury only ever shows money that was really earned, so a
      // dry-run village sits at $0.00 no matter how much it produces.
      el.amount.classList.toggle('is-zero', !stats.monthRevenueCents);

      el.chipMode.textContent = stats.dryRun ? 'DRY RUN' : 'LIVE';
      el.chipMode.style.background = stats.dryRun ? 'var(--brass)' : 'var(--moss)';
      el.chipMode.title = stats.dryRun
        ? 'Simulated: nothing reaches Printify, Etsy or YouTube, and nothing counts as revenue'
        : 'Live: listings publish for real';

      this.renderLedger(stats);
      this.animateRevenue(Number(stats.monthRevenue) || 0);
    },

    /**
     * The per-channel breakdown, plus the little icon+amount strip that sits
     * under the total so the split is legible without opening anything.
     */
    renderLedger(stats) {
      const channels = stats.channels || [];
      if (channels.length === 0) return;

      el.sparks.innerHTML = channels
        .map((channel) => {
          // A channel with only simulated money shows its icon greyed with a
          // dash, never a figure — a number here would read as earnings.
          const simulatedOnly = channel.monthCents === 0 && channel.simulatedCents > 0;
          const title = simulatedOnly
            ? `${channel.label}: no real revenue (${this.money(channel.simulated)} simulated)`
            : `${channel.label}: ${this.money(channel.month)}`;
          return (
            `<span class="spark${simulatedOnly ? ' spark--sim' : ''}" title="${esc(title)}">` +
            `${channel.icon}<span>${simulatedOnly ? '—' : this.money(channel.month)}</span></span>`
          );
        })
        .join('');

      el.ledgerRows.innerHTML = channels
        .map((channel) => {
          const detail = channel.detail ? `<span class="ledger__detail">${esc(channel.detail)}</span>` : '';
          const flag = channel.estimated
            ? '<span class="ledger__est" title="Projected, not money received">est</span>'
            : '';
          const sim = channel.simulatedCents
            ? `<span class="ledger__sim" title="Dry-run activity, deliberately not counted">` +
              `${this.money(channel.simulated)} sim</span>`
            : '';
          return (
            `<li class="ledger__row" style="--accent:${esc(channel.accent)}">` +
            `<span class="ledger__name"><span class="ledger__icon">${channel.icon}</span>` +
            `<span><span class="ledger__label">${esc(channel.label)}</span>${detail}</span></span>` +
            `<span class="ledger__amount">${this.money(channel.month)}${flag}</span>` +
            `<span class="ledger__amount ledger__amount--dim">${this.money(channel.lifetime)}${sim}</span>` +
            `</li>`
          );
        })
        .join('');

      const notes = [];
      if (stats.simulatedRevenueCents > 0 && !stats.countsDryRun) {
        notes.push(
          `${this.money(stats.simulatedRevenue)} of dry-run activity is recorded but NOT counted — ` +
            'nothing was actually sold.',
        );
      }
      if (channels.some((channel) => channel.estimated)) {
        notes.push('YouTube figures are estimates: views × RPM ÷ 1000, not confirmed payouts.');
      }
      if (notes.length === 0) {
        notes.push('Every line is real, settled revenue.');
      }
      el.ledgerNote.textContent = notes.join(' ');
    },

    toggleLedger(force) {
      const open = force === undefined ? el.ledger.hidden : force;
      el.ledger.hidden = !open;
      el.treasuryToggle.setAttribute('aria-expanded', String(open));
      el.treasuryToggle.classList.toggle('is-open', open);
    },

    /**
     * Roll the gold counter up rather than snapping it — a jackpot tick is the
     * whole point of a treasury widget.
     */
    animateRevenue(target) {
      const from = this.displayedRevenue;
      if (Math.abs(target - from) < 0.005) {
        el.amount.textContent = this.money(target);
        return;
      }
      if (target > from) window.Sfx.coin();

      const started = performance.now();
      const duration = 700;
      const step = (now) => {
        const t = Math.min(1, (now - started) / duration);
        // Ease-out so it decelerates into the final figure.
        const eased = 1 - Math.pow(1 - t, 3);
        const value = from + (target - from) * eased;
        el.amount.textContent = this.money(value);
        if (t < 1) requestAnimationFrame(step);
        else this.displayedRevenue = target;
      };
      requestAnimationFrame(step);
    },

    money(value) {
      return '$' + Number(value || 0).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    },

    /* ------------------------------------------------------ connection */

    setConnection(state) {
      const map = {
        connected: ['● live', 'chip chip--live'],
        connecting: ['◌ connecting', 'chip chip--muted'],
        disconnected: ['✕ offline', 'chip chip--down'],
      };
      const [text, className] = map[state] || map.connecting;
      el.chipLink.textContent = text;
      el.chipLink.className = className;
    },

    /* ------------------------------------------------------- chronicle */

    log(message, tone = 'info') {
      const item = document.createElement('li');
      const time = new Date().toLocaleTimeString([], { hour12: false });
      item.innerHTML =
        `<time>${esc(time)}</time><span class="tone-${esc(tone)}">${esc(message)}</span>`;
      el.log.appendChild(item);

      while (el.log.children.length > MAX_LOG) el.log.removeChild(el.log.firstChild);
      el.log.scrollTop = el.log.scrollHeight;
    },

    clearLog() {
      el.log.innerHTML = '';
    },

    /* -------------------------------------------------------- listings */

    renderListings(listings) {
      el.listings.innerHTML = '';
      el.listingsCount.textContent = (listings && listings.length) || 0;
      if (!listings || listings.length === 0) {
        el.listings.innerHTML =
          '<li class="listing-card"><p class="listing-card__meta">No listings yet — press Run Village.</p></li>';
        return;
      }
      listings.forEach((listing) => el.listings.appendChild(this.listingCard(listing)));
    },

    /** Insert or replace one listing card, newest first. */
    upsertListing(listing) {
      if (!listing || !listing.id) return;
      const existing = el.listings.querySelector(`[data-listing="${listing.id}"]`);
      const card = this.listingCard(listing);
      if (existing) {
        existing.replaceWith(card);
      } else {
        const placeholder = el.listings.querySelector('.listing-card:not([data-listing])');
        if (placeholder) placeholder.remove();
        el.listings.prepend(card);
        while (el.listings.children.length > 30) el.listings.removeChild(el.listings.lastChild);
      }
    },

    listingCard(listing) {
      const item = document.createElement('li');
      item.className = 'listing-card';
      item.dataset.listing = listing.id;

      const colours = {
        PUBLISHED: '#1d4a24',
        PENDING_APPROVAL: '#4a3a12',
        FAILED: '#4a1620',
        REJECTED: '#3a1a2a',
        APPROVED: '#14304a',
        DRAFTED: '#2a3346',
      };

      const price = listing.price_cents ? this.money(listing.price_cents / 100) : '';
      item.innerHTML =
        `<div class="listing-card__head">` +
        `<span class="listing-card__id">#${esc(listing.id)}</span>` +
        `<span class="listing-card__status" style="background:${colours[listing.status] || '#2a3346'}">${esc(listing.status)}</span>` +
        `</div>` +
        `<p class="listing-card__title">${esc(listing.title || listing.niche || 'untitled')}</p>` +
        `<p class="listing-card__meta">${esc(listing.niche || '')} · ${esc(price)}${listing.dry_run ? ' · dry-run' : ''}</p>`;

      if (listing.status === 'PENDING_APPROVAL') {
        const actions = document.createElement('div');
        actions.className = 'listing-card__actions';

        const approve = document.createElement('button');
        approve.type = 'button';
        approve.className = 'btn btn--primary';
        approve.textContent = '✅ Approve';
        approve.addEventListener('click', () => {
          window.Sfx.confirm();
          if (this.onListingAction) this.onListingAction('approve', listing.id);
        });

        const reject = document.createElement('button');
        reject.type = 'button';
        reject.className = 'btn btn--danger';
        reject.textContent = '🚫 Reject';
        reject.addEventListener('click', () => {
          window.Sfx.error();
          if (this.onListingAction) this.onListingAction('reject', listing.id);
        });

        actions.append(approve, reject);
        item.appendChild(actions);
      }

      return item;
    },

    /* -------------------------------------------------- Bard's Cinema */

    /** Newest first. Each card is a real player, not a thumbnail. */
    renderShorts(shorts) {
      this.shorts = shorts || [];
      const count = this.shorts.length;

      el.cinemaCount.hidden = count === 0;
      el.cinemaCount.textContent = count;

      const stats = (this.stats && this.stats.shorts) || {};
      el.cinemaStats.innerHTML =
        `<span class="cinema__stat">🎬 <b>${stats.total || count}</b> made</span>` +
        `<span class="cinema__stat">📡 <b>${stats.published || 0}</b> live</span>` +
        `<span class="cinema__stat">👁 <b>${(stats.views || 0).toLocaleString()}</b> views</span>` +
        `<span class="cinema__stat">💵 <b>${(stats.rpmCents || 0).toFixed(0)}c</b> RPM</span>`;

      el.shorts.innerHTML = '';
      if (count === 0) {
        el.shorts.innerHTML =
          '<li class="short-card short-card--empty">' +
          '<p>The Bard has written nothing yet.</p>' +
          '<p class="short-card__hint">Press <strong>+ new</strong>, or poke Finneas at the Theater.</p>' +
          '</li>';
        return;
      }
      this.shorts.forEach((short) => el.shorts.appendChild(this.shortCard(short)));
    },

    shortCard(short) {
      const item = document.createElement('li');
      item.className = 'short-card';
      item.dataset.short = short.id;

      const storyboard = short.render_backend === 'storyboard';
      const silent = short.voice_backend === 'placeholder';
      const ready = Boolean(short.video_path);
      const runtime = Math.round(short.duration_seconds || 0);

      const media = !ready
        ? '<div class="short-card__media short-card__media--pending">rendering…</div>'
        : storyboard
          ? `<img class="short-card__media" src="/api/shorts/${short.id}/video" alt="Storyboard" loading="lazy" />`
          : `<video class="short-card__media" src="/api/shorts/${short.id}/video" controls
               playsinline preload="none"
               ${short.thumbnail_path ? `poster="/api/shorts/${short.id}/thumbnail"` : ''}></video>`;

      const warnings = [];
      if (storyboard) warnings.push('storyboard only');
      if (silent) warnings.push('silent track');

      item.innerHTML =
        media +
        `<div class="short-card__body">` +
        `<div class="short-card__head">` +
        `<span class="short-card__id">#${esc(short.id)}</span>` +
        `<span class="short-card__status" data-status="${esc(short.status)}">${esc(short.status)}</span>` +
        `</div>` +
        `<p class="short-card__title">${esc(short.title || short.topic || 'untitled')}</p>` +
        `<p class="short-card__meta">${esc(short.category || '')} · ${runtime}s` +
        `${short.dry_run ? ' · dry-run' : ''}</p>` +
        (short.views
          ? `<p class="short-card__perf">👁 ${short.views.toLocaleString()} · ` +
            `${this.money((short.estimated_cents || 0) / 100)} est</p>`
          : '') +
        (warnings.length ? `<p class="short-card__warn">⚠️ ${esc(warnings.join(' · '))}</p>` : '') +
        (short.youtube_url
          ? `<p class="short-card__live">▶ <a href="${esc(short.youtube_url)}" target="_blank" ` +
            `rel="noopener noreferrer">watch on YouTube</a>` +
            // A private upload is published as far as this pipeline knows, but
            // nobody can see it yet — say so rather than implying it is live.
            `${short.youtube_privacy && short.youtube_privacy !== 'public'
                ? ` <span class="short-card__privacy">${esc(short.youtube_privacy)}</span>` : ''}</p>`
          : '') +
        `</div>`;

      const actions = document.createElement('div');
      actions.className = 'short-card__actions';

      if (short.status === 'PENDING_APPROVAL') {
        actions.appendChild(
          this.smallButton('✅ Approve', 'btn--primary', () => this.emitShort('approve', short.id)),
        );
        actions.appendChild(
          this.smallButton('🔀 Reroll', '', () => this.emitShort('reroll', short.id)),
        );
        actions.appendChild(
          this.smallButton('🚫', 'btn--danger', () => this.emitShort('reject', short.id)),
        );
      } else if (short.status === 'PUBLISHED') {
        // Once it is live, the useful action is recording how it performed.
        actions.appendChild(
          this.smallButton('👁 Log views', '', () => this.promptViews(short)),
        );
      }

      if (actions.children.length > 0) item.appendChild(actions);
      return item;
    },

    smallButton(label, extra, onClick) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn ' + extra;
      button.textContent = label;
      button.addEventListener('click', () => {
        window.Sfx.select();
        onClick();
      });
      return button;
    },

    /** Ask for a view count, then post it so the treasury re-estimates. */
    promptViews(short) {
      const answer = window.prompt(
        `Views for "${short.title}"?\n\n` +
          'This is entered by hand — there is no YouTube API connected. ' +
          'The estimate is views ÷ 1000 × RPM.',
        String(short.views || 0),
      );
      if (answer === null) return;
      const views = Number.parseInt(answer, 10);
      if (!Number.isFinite(views) || views < 0) {
        this.toast('That is not a view count.', 'warn');
        return;
      }
      if (this.onShortAction) this.onShortAction('metrics', short.id, { views });
    },

    emitShort(action, shortId) {
      if (this.onShortAction) this.onShortAction(action, shortId);
    },

    /* ------------------------------------------------------------ deals */

    /** Money, from cents, without trusting the locale to add a symbol. */
    dealMoney(cents) {
      return '$' + (Number(cents || 0) / 100).toFixed(2);
    },

    renderDeals(payload) {
      const deals = (payload && payload.deals) || [];
      const metrics = (payload && payload.metrics) || {};
      this.deals = deals;

      if (el.dealsCount) {
        el.dealsCount.textContent = String(deals.length);
        el.dealsCount.hidden = deals.length === 0;
      }

      if (el.dealsStats) {
        // Clicks and earnings are only ever what someone typed in from the
        // Associates dashboard, so they are labelled as reported rather than
        // presented as though this app measured them.
        const clicks = Number(metrics.clicks || 0);
        el.dealsStats.innerHTML =
          `<span title="curated recommendations">🛒 ${Number(metrics.curated || 0)}</span>` +
          `<span title="approved and ready to post">📣 ${Number(metrics.published || 0)}</span>` +
          `<span title="clicks, as reported to you by Amazon">👆 ${clicks.toLocaleString()}</span>` +
          `<span title="earnings, as reported to you by Amazon">${this.dealMoney(metrics.earningsCents)}</span>` +
          (clicks
            ? `<span title="earnings per click">EPC ${this.dealMoney(
                (metrics.earningsPerClick || 0) * 100,
              )}</span>`
            : '');
      }

      if (!el.deals) return;
      el.deals.innerHTML = '';
      if (!deals.length) {
        const empty = document.createElement('li');
        empty.className = 'rail__empty';
        empty.textContent = 'No deals yet — press + scout.';
        el.deals.appendChild(empty);
        return;
      }
      deals.forEach((deal) => el.deals.appendChild(this.dealCard(deal)));
    },

    dealCard(deal) {
      const item = document.createElement('li');
      item.className = 'deal-card';
      item.dataset.deal = deal.id;

      const payload = deal.payload || {};
      const low = Number(deal.price_low || 0);
      const high = Number(deal.price_high || 0);
      const price =
        low && high && low !== high
          ? `~$${low.toFixed(0)}-$${high.toFixed(0)}`
          : high || low
            ? `~$${(high || low).toFixed(0)}`
            : 'price varies';

      const cons = Array.isArray(payload.cons) ? payload.cons : [];

      item.innerHTML =
        `<div class="deal-card__head">` +
        `<span class="deal-card__id">#${esc(deal.id)}</span>` +
        `<span class="deal-card__status" data-status="${esc(deal.status)}">${esc(
          deal.status.toLowerCase(),
        )}</span>` +
        `</div>` +
        `<p class="deal-card__hook">${esc(deal.hook || payload.hook || '')}</p>` +
        `<p class="deal-card__product">${esc(deal.product)}</p>` +
        `<p class="deal-card__meta">${esc(deal.category || '')} · ${esc(price)}` +
        `${deal.dry_run ? ' · dry-run' : ''}</p>` +
        // The trade-offs are the reason the recommendation is worth anything;
        // showing only upside would make the card an advert.
        (cons.length
          ? `<p class="deal-card__cons">👎 ${esc(cons[0])}</p>`
          : '') +
        (deal.clicks
          ? `<p class="deal-card__perf">👆 ${Number(deal.clicks).toLocaleString()} · ` +
            `${this.dealMoney(deal.earnings_cents)} · EPC ${this.dealMoney(
              (deal.earnings_per_click || 0) * 100,
            )}</p>`
          : '') +
        (deal.affiliate_url
          ? `<p class="deal-card__link"><a href="${esc(deal.affiliate_url)}" ` +
            `target="_blank" rel="noopener noreferrer sponsored">🔗 Amazon</a></p>`
          : '');

      const actions = document.createElement('div');
      actions.className = 'deal-card__actions';

      if (deal.status === 'DRAFTED' || deal.status === 'PENDING_APPROVAL') {
        actions.appendChild(
          this.smallButton('✅ Approve & Post', 'btn--primary', () =>
            this.emitDeal('approve', deal.id),
          ),
        );
        actions.appendChild(
          this.smallButton('❌', 'btn--danger', () => this.emitDeal('reject', deal.id)),
        );
      } else if (deal.status === 'PUBLISHED') {
        actions.appendChild(
          this.smallButton('📊 Log clicks', '', () => this.promptDealMetrics(deal)),
        );
      }

      if (actions.childElementCount) item.appendChild(actions);
      return item;
    },

    promptDealMetrics(deal) {
      const clicks = window.prompt(
        `Clicks reported by Amazon for #${deal.id}:`,
        String(deal.clicks || 0),
      );
      if (clicks === null) return;
      const earnings = window.prompt(
        `Earnings in dollars for #${deal.id}:`,
        ((deal.earnings_cents || 0) / 100).toFixed(2),
      );
      if (earnings === null) return;

      const cents = Math.round(Number(earnings) * 100);
      if (!Number.isFinite(Number(clicks)) || !Number.isFinite(cents)) {
        this.toast('Clicks and earnings must be numbers', 'warn');
        return;
      }
      if (this.onDealMetrics) {
        this.onDealMetrics(deal.id, {
          clicks: Number(clicks),
          earnings_cents: cents,
        });
      }
    },

    emitDeal(action, dealId) {
      if (this.onDealAction) this.onDealAction(action, dealId);
    },

    /* ------------------------------------------------------------- tabs */

    selectTab(name) {
      el.tabs.forEach((tab) => {
        const selected = tab.dataset.panel === name;
        tab.classList.toggle('is-selected', selected);
        tab.setAttribute('aria-selected', String(selected));
      });
      Object.entries(el.panels).forEach(([key, panel]) => {
        if (!panel) return;
        panel.hidden = key !== name;
        panel.classList.toggle('is-active', key === name);
      });
    },

    /* ---------------------------------------------------------- toasts */

    toast(message, tone = 'info') {
      const node = document.createElement('div');
      node.className = 'toast toast--' + tone;
      node.textContent = message;
      el.toasts.appendChild(node);
      window.setTimeout(() => {
        node.style.transition = 'opacity .3s';
        node.style.opacity = '0';
        window.setTimeout(() => node.remove(), 320);
      }, 4200);
    },
  };

  el.clearLog.addEventListener('click', () => Hud.clearLog());

  el.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      window.Sfx.select();
      Hud.selectTab(tab.dataset.panel);
    });
  });

  el.newShort.addEventListener('click', () => {
    window.Sfx.confirm();
    if (Hud.onGenerateShort) Hud.onGenerateShort();
  });

  if (el.newDeal) {
    el.newDeal.addEventListener('click', () => {
      window.Sfx.confirm();
      if (Hud.onScoutDeal) Hud.onScoutDeal();
    });
  }

  el.treasuryToggle.addEventListener('click', () => {
    window.Sfx.select();
    Hud.toggleLedger();
  });

  // Clicking away closes the ledger, the way a real dropdown behaves.
  document.addEventListener('click', (event) => {
    if (el.ledger.hidden) return;
    if (!document.getElementById('treasury').contains(event.target)) Hud.toggleLedger(false);
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !el.ledger.hidden) Hud.toggleLedger(false);
  });

  window.Hud = Hud;
})();

/* ==========================================================================
   dialogue.js — retro RPG dialogue box, typewriter text, and sound cues.
   Exposes window.Dialogue and window.Sfx. No dependencies.
   ========================================================================== */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ Sfx */

  /**
   * Chiptune blips synthesised with WebAudio. No audio files to ship, and no
   * sound at all until the first user gesture, because browsers block audio
   * that starts on its own.
   */
  const Sfx = {
    ctx: null,
    muted: false,

    /** Lazily create the context; safe to call on every cue. */
    ensure() {
      if (this.muted) return null;
      if (!this.ctx) {
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (!Ctor) return null;
        this.ctx = new Ctor();
      }
      if (this.ctx.state === 'suspended') this.ctx.resume();
      return this.ctx;
    },

    /**
     * One square-wave note.
     * @param {number} freq Hz
     * @param {number} duration seconds
     * @param {number} volume 0-1
     * @param {OscillatorType} type
     */
    blip(freq, duration = 0.05, volume = 0.05, type = 'square') {
      const ctx = this.ensure();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type;
      osc.frequency.value = freq;
      // A short attack/decay envelope; a bare gate would click.
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(volume, ctx.currentTime + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration + 0.02);
    },

    /** Played per character while the typewriter runs. */
    type() {
      this.blip(620 + Math.random() * 120, 0.022, 0.022);
    },

    open() {
      this.blip(392, 0.06, 0.05);
      setTimeout(() => this.blip(587, 0.09, 0.05), 60);
    },

    close() {
      this.blip(392, 0.06, 0.04);
      setTimeout(() => this.blip(262, 0.09, 0.04), 55);
    },

    select() {
      this.blip(880, 0.05, 0.05);
    },

    confirm() {
      this.blip(523, 0.05, 0.05);
      setTimeout(() => this.blip(659, 0.05, 0.05), 55);
      setTimeout(() => this.blip(784, 0.11, 0.05), 110);
    },

    error() {
      this.blip(180, 0.16, 0.06, 'sawtooth');
    },

    coin() {
      this.blip(988, 0.05, 0.045);
      setTimeout(() => this.blip(1319, 0.16, 0.045), 55);
    },

    toggle() {
      this.muted = !this.muted;
      if (!this.muted) this.select();
      return this.muted;
    },
  };

  /* -------------------------------------------------------------- Dialogue */

  const el = {
    backdrop: document.getElementById('dialogue-backdrop'),
    box: document.getElementById('dialogue'),
    close: document.getElementById('dialogue-close'),
    name: document.getElementById('dialogue-name'),
    title: document.getElementById('dialogue-title'),
    state: document.getElementById('dialogue-state'),
    dot: document.getElementById('dialogue-dot'),
    text: document.getElementById('dialogue-text'),
    outputWrap: document.getElementById('dialogue-output'),
    preview: document.getElementById('dialogue-preview'),
    actions: document.getElementById('dialogue-actions'),
    portrait: document.getElementById('portrait-canvas'),
  };

  /** Escape text before it goes anywhere near innerHTML. */
  function esc(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  const Dialogue = {
    open: false,
    subject: null,
    _timer: null,
    _full: '',
    _onAction: null,

    /**
     * Show the box.
     * @param {Object} options
     * @param {string} options.name
     * @param {string} options.title
     * @param {string} options.status idle|working|done|error
     * @param {string} options.text the line to type out
     * @param {Object|null} options.output last produced asset
     * @param {Array} options.actions [{id,label,description,slow}]
     * @param {Function} options.onAction called with an action id
     * @param {Function|null} options.drawPortrait receives the portrait ctx
     * @param {string} options.subjectId
     */
    show(options) {
      this.subject = options.subjectId || null;
      this._onAction = options.onAction || null;

      el.name.textContent = options.name || '';
      el.title.textContent = options.title || '';
      el.state.textContent = options.status || 'idle';
      el.dot.className = 'dot status-' + (options.status || 'idle');

      this.renderPortrait(options.drawPortrait);
      this.renderOutput(options.output, options.html);
      this.renderActions(options.actions || []);

      el.backdrop.hidden = false;
      this.open = true;
      Sfx.open();
      this.type(options.text || '');
    },

    /** Re-render an already-open box without replaying the open cue. */
    update(options) {
      if (!this.open || this.subject !== options.subjectId) return;
      el.state.textContent = options.status || 'idle';
      el.dot.className = 'dot status-' + (options.status || 'idle');
      this.renderOutput(options.output, options.html);
      if (options.text && options.text !== this._full) {
        this.type(options.text);
      }
    },

    /** Typewriter effect with a per-character blip. */
    type(text) {
      window.clearInterval(this._timer);
      this._full = String(text || '');
      el.text.textContent = '';
      el.text.classList.add('is-typing');

      let index = 0;
      const speed = 16;
      this._timer = window.setInterval(() => {
        if (index >= this._full.length) {
          window.clearInterval(this._timer);
          el.text.classList.remove('is-typing');
          return;
        }
        el.text.textContent += this._full[index];
        // Blip on roughly every third glyph; every one is a machine-gun.
        if (index % 3 === 0 && this._full[index] !== ' ') Sfx.type();
        index += 1;
      }, speed);
    },

    /** Reveal the whole line at once — click-to-skip. */
    finishTyping() {
      if (!this._timer) return false;
      window.clearInterval(this._timer);
      this._timer = null;
      if (el.text.textContent !== this._full) {
        el.text.textContent = this._full;
        el.text.classList.remove('is-typing');
        return true;
      }
      el.text.classList.remove('is-typing');
      return false;
    },

    renderPortrait(draw) {
      const ctx = el.portrait.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, 64, 64);
      ctx.fillStyle = '#0d1119';
      ctx.fillRect(0, 0, 64, 64);
      if (typeof draw === 'function') draw(ctx, 64);
    },

    /**
     * Build the "last output" preview card.
     *
     * @param {Object|null} output the agent's typed last-output record
     * @param {string|null} html a pre-built card, used by agents whose output
     *   is richer than a summary — the Bard's playable video, for one. It is
     *   composed by the caller from already-escaped values.
     */
    renderOutput(output, html) {
      if (html) {
        el.outputWrap.hidden = false;
        el.preview.innerHTML = html;
        return;
      }
      if (!output) {
        el.outputWrap.hidden = true;
        el.preview.innerHTML = '';
        return;
      }
      el.outputWrap.hidden = false;

      const kind = output.kind || 'output';
      let thumb = '';
      let body = '';

      if (output.image_url) {
        thumb = `<img class="preview__thumb" src="${esc(output.image_url)}" alt="Latest artwork" />`;
      }

      if (kind === 'brief') {
        body =
          `<p class="preview__line"><strong>${esc(output.niche)}</strong></p>` +
          `<p class="preview__line preview__muted">${esc(output.audience)}</p>` +
          `<p class="preview__line preview__muted">${esc((output.prompt || '').slice(0, 200))}…</p>`;
      } else if (kind === 'artwork' || kind === 'print_file') {
        const size = output.width ? `${output.width}×${output.height} @ ${output.dpi} DPI` : '';
        body =
          `<p class="preview__line">${esc(output.simulated ? 'Simulated artwork' : 'Generated artwork')}</p>` +
          `<p class="preview__line preview__muted">${esc(output.model || size)}</p>` +
          (size ? `<p class="preview__line preview__muted">${esc(size)}</p>` : '') +
          (output.background_removed !== undefined
            ? `<p class="preview__line preview__muted">background ${output.background_removed ? 'removed' : 'kept'}</p>`
            : '');
      } else if (kind === 'copy') {
        const tags = (output.tags || [])
          .map((tag) => `<span class="preview__tag">${esc(tag)}</span>`)
          .join('');
        body =
          `<p class="preview__line"><strong>${esc(output.title)}</strong></p>` +
          `<div class="preview__tags">${tags}</div>` +
          `<p class="preview__line preview__muted">${esc((output.description || '').slice(0, 140))}…</p>`;
      } else if (kind === 'screen') {
        const findings = [...(output.errors || []), ...(output.warnings || [])].slice(0, 3);
        body =
          `<p class="preview__line">${output.ok ? '✅ ' : '⛔ '}${esc(output.summary)}</p>` +
          findings.map((f) => `<p class="preview__line preview__muted">• ${esc(f)}</p>`).join('');
      } else if (kind === 'dispatch') {
        body =
          `<p class="preview__line">${output.delivered ? '🔔 Sent to Telegram' : '🔕 Telegram not configured'}</p>` +
          `<p class="preview__line preview__muted">Listing #${esc(output.listing_id)} — ${esc(output.title)}</p>`;
      } else if (kind === 'publish') {
        body =
          `<p class="preview__line">${output.ok ? '📦 Published' : '⚠️ Failed'}${output.simulated ? ' (simulated)' : ''}</p>` +
          `<p class="preview__line preview__muted">${esc(output.product_id || output.error || '')}</p>`;
      } else {
        body = `<p class="preview__line preview__muted">${esc(JSON.stringify(output).slice(0, 200))}</p>`;
      }

      el.preview.innerHTML =
        thumb + `<div class="preview__body"><p class="preview__kind">${esc(kind)}</p>${body}</div>`;
    },

    renderActions(actions) {
      el.actions.innerHTML = '';
      actions.forEach((action) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn' + (action.slow ? ' btn--primary' : '');
        button.textContent = action.label;
        button.title = action.description || '';
        button.addEventListener('click', (event) => {
          event.stopPropagation();
          Sfx.confirm();
          if (this._onAction) this._onAction(action.id, action);
        });
        el.actions.appendChild(button);
      });
    },

    hide() {
      if (!this.open) return;
      window.clearInterval(this._timer);
      this._timer = null;
      el.backdrop.hidden = true;
      this.open = false;
      this.subject = null;
      Sfx.close();
    },
  };

  /* ------------------------------------------------------------- wiring */

  el.close.addEventListener('click', (event) => {
    event.stopPropagation();
    Dialogue.hide();
  });

  // Click the backdrop to dismiss; click the box to skip the typewriter.
  el.backdrop.addEventListener('click', (event) => {
    if (event.target === el.backdrop) {
      Dialogue.hide();
    } else if (!Dialogue.finishTyping()) {
      // Already fully typed and the click was not on a control: leave it open.
    }
  });

  document.addEventListener('keydown', (event) => {
    if (!Dialogue.open) return;
    if (event.key === 'Escape') Dialogue.hide();
    if (event.key === 'Enter' || event.key === ' ') Dialogue.finishTyping();
  });

  window.Dialogue = Dialogue;
  window.Sfx = Sfx;
  window.escapeHtml = esc;
})();

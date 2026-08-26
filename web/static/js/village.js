/* ==========================================================================
   village.js — the living 16-bit village.

   Owns the tile map, the buildings, the seven villager sprites, their
   pathfinding and animation, the camera, hit-testing for click-to-talk, and
   the WebSocket that drives all of it from the Python pipeline.

   Everything is drawn procedurally into a fixed 960x600 buffer and scaled with
   nearest-neighbour sampling, so there are no image assets to ship and the art
   stays crisp at any window size.
   ========================================================================== */
(function () {
  'use strict';

  const TILE = 24;
  const COLS = 40;
  const ROWS = 25;
  const WORLD_W = COLS * TILE; // 960
  const WORLD_H = ROWS * TILE; // 600

  const canvas = document.getElementById('village');
  const ctx = canvas.getContext('2d', { alpha: false });
  const nametagLayer = document.getElementById('nametags');

  /* ======================================================================
     Palette — warm, earthen, and deliberately limited.

     Every ramp is three or four steps that share a hue and drift warmer as
     they lighten, which is what stops 16-bit art reading as flat vector fills.
     Greens lean olive rather than emerald; stone leans sandstone rather than
     blue-grey; every shadow carries a little red.
     ====================================================================== */

  const P = {
    // ---- grass: olive-warm, four steps -------------------------------
    grassDeep: '#38571f',
    grassDark: '#456a26',
    grass: '#57802f',
    grassLit: '#6d9a3c',
    grassHi: '#87b04b',
    grassDry: '#8a8a3a',

    // ---- earth and cobble --------------------------------------------
    dirtDeep: '#4a3320',
    dirt: '#6b4a2c',
    dirtLit: '#8a6440',
    cobbleDark: '#6d5a44',
    cobble: '#947d5f',
    cobbleLit: '#b39a78',
    cobbleHi: '#cbb28d',
    mortar: '#54432f',
    sand: '#c4a570',
    sandDark: '#a2854f',

    // ---- water: warm teal, not cold blue ------------------------------
    waterDeep: '#1f4a56',
    water: '#2d6b72',
    waterLit: '#3f8a86',
    waterFoam: '#a8d4c0',

    // ---- timber --------------------------------------------------------
    woodDeep: '#3a2413',
    woodDark: '#54341c',
    wood: '#7a5230',
    woodLit: '#9a6c42',
    woodHi: '#b98a58',

    // ---- plaster and stone ---------------------------------------------
    plasterDark: '#b09472',
    plaster: '#d6bd97',
    plasterLit: '#e8d5b0',
    stoneDeep: '#4a4038',
    stoneDark: '#6b5f52',
    stone: '#8d7f6e',
    stoneLit: '#ab9c88',
    stoneHi: '#c6b7a0',

    // ---- roofing: slate that leans plum, and warm thatch ---------------
    slateDeep: '#3a3038',
    slateDark: '#4e414c',
    slate: '#645361',
    slateLit: '#7d6a78',
    thatchDark: '#8a6a2f',
    thatch: '#b08e42',
    thatchLit: '#cfae5c',
    tileRedDeep: '#5e2418',
    tileRedDark: '#7c3423',
    tileRed: '#9e4630',
    tileRedLit: '#bd5f42',
    tileGreenDark: '#2f4a2c',
    tileGreen: '#456b3e',
    tileGreenLit: '#5c8a50',
    tilePlumDark: '#452f4a',
    tilePlum: '#5e4266',
    tilePlumLit: '#7a5a82',

    // ---- foliage --------------------------------------------------------
    leafDeep: '#22401d',
    leafDark: '#2f5426',
    leaf: '#3f6d31',
    leafLit: '#568a3f',
    leafHi: '#72a854',
    trunkDark: '#3a2616',
    trunk: '#573823',
    trunkLit: '#75503a',

    // ---- light and accents ----------------------------------------------
    outline: '#241a12',
    outlineSoft: '#3a2a1c',
    windowLit: '#ffcf6b',
    windowGlow: '#ffe6a8',
    windowDark: '#2e2418',
    ember: '#e8763a',
    emberHi: '#ffb35c',
    gold: '#e0a53f',
    goldHi: '#f5cd6a',
    cloth: '#b4472f',
    shadow: 'rgba(40, 26, 14, 0.30)',
    shadowSoft: 'rgba(40, 26, 14, 0.16)',

    // ---- skin ramp -------------------------------------------------------
    skinHi: '#f4d3ac',
    skin: '#e0b184',
    skinMid: '#c08d5e',
    skinDark: '#8e6039',
    white: '#f2e6c8',
  };

  /* ======================================================================
     Texture baking

     Ground tiles are painted ONCE into offscreen canvases at boot and blitted
     every frame. That buys two things: real per-pixel texture (clusters of
     grass blades, individually shaped cobbles, ripple bands) instead of a flat
     fill, and a draw cost of one drawImage per tile rather than a few dozen
     fillRects.

     Several variants per material are baked and chosen by a deterministic hash
     of the tile coordinate, so the ground never tiles visibly and never
     shimmers between frames.
     ====================================================================== */

  const TEX = { grass: [], path: [], water: [], dirt: [] };

  /**
   * Paint one tile-sized offscreen canvas.
   * @param {(g: CanvasRenderingContext2D, size: number) => void} painter
   * @returns {HTMLCanvasElement}
   */
  function bakeTile(painter, size) {
    const tile = document.createElement('canvas');
    tile.width = size || TILE;
    tile.height = size || TILE;
    const g = tile.getContext('2d');
    g.imageSmoothingEnabled = false;
    painter(g, tile.width);
    return tile;
  }

  /** Seeded RNG so every bake is identical run to run. */
  function seeded(seed) {
    let value = seed * 16807 % 2147483647;
    return () => {
      value = (value * 16807) % 2147483647;
      return (value - 1) / 2147483646;
    };
  }

  const dot = (g, x, y, w, h, colour) => {
    g.fillStyle = colour;
    g.fillRect(x, y, w, h);
  };

  /**
   * Grass: a dithered two-tone base, then blade clusters and the occasional
   * dry tuft or tiny flower. The dither is what gives it the "woven" look a
   * flat fill can never have.
   */
  function bakeGrass(variant) {
    return bakeTile((g, size) => {
      const random = seeded(variant * 977 + 13);

      // Base with a checkerboard dither between two adjacent greens.
      for (let y = 0; y < size; y += 2) {
        for (let x = 0; x < size; x += 2) {
          const roll = random();
          dot(g, x, y, 2, 2, roll > 0.55 ? P.grass : P.grassDark);
        }
      }
      // Scatter a lighter dither on top so the surface is not uniform.
      for (let i = 0; i < 26; i += 1) {
        const x = Math.floor(random() * size);
        const y = Math.floor(random() * size);
        dot(g, x, y, 1, 1, P.grassLit);
      }
      // Blade clusters: three pixels rising, lit on the tip.
      const blades = 4 + Math.floor(random() * 3);
      for (let i = 0; i < blades; i += 1) {
        const x = 2 + Math.floor(random() * (size - 5));
        const y = 3 + Math.floor(random() * (size - 7));
        dot(g, x, y, 1, 3, P.grassDeep);
        dot(g, x, y, 1, 2, P.grassLit);
        dot(g, x + 1, y + 1, 1, 2, P.grassDark);
        dot(g, x - 1, y + 2, 1, 2, P.grassLit);
        dot(g, x, y - 1, 1, 1, P.grassHi);
      }
      // A dry tuft or a flower head, rarely.
      if (variant % 3 === 0) {
        const x = 4 + Math.floor(random() * (size - 9));
        const y = 6 + Math.floor(random() * (size - 12));
        dot(g, x, y, 3, 1, P.grassDry);
        dot(g, x + 1, y - 1, 1, 1, P.grassDry);
      }
      if (variant === 3) {
        const x = 6 + Math.floor(random() * (size - 12));
        const y = 8 + Math.floor(random() * (size - 14));
        dot(g, x, y, 2, 2, '#d8c04a');
        dot(g, x, y, 1, 1, '#f0dc72');
        dot(g, x, y + 2, 1, 2, P.grassDeep);
      }
    });
  }

  /**
   * Cobblestone: irregular rounded stones separated by dark mortar, each with
   * a lit top edge and a shadowed base, laid in offset rows.
   */
  function bakePath(variant) {
    return bakeTile((g, size) => {
      const random = seeded(variant * 613 + 41);

      // Mortar bed, dithered so the gaps are not a flat colour.
      for (let y = 0; y < size; y += 2) {
        for (let x = 0; x < size; x += 2) {
          dot(g, x, y, 2, 2, random() > 0.5 ? P.mortar : P.dirtDeep);
        }
      }

      // Stones in two offset rows of three, with jittered size.
      const rows = [
        { y: 1, offset: 0 },
        { y: 9, offset: 4 },
        { y: 17, offset: 1 },
      ];
      rows.forEach((row, rowIndex) => {
        for (let i = -1; i < 4; i += 1) {
          const w = 5 + Math.floor(random() * 3);
          const h = 5 + Math.floor(random() * 2);
          const x = row.offset + i * 8 + Math.floor(random() * 2);
          const y = row.y + Math.floor(random() * 2);
          if (x > size || x + w < 0) continue;

          const tone = random();
          const body = tone > 0.72 ? P.cobbleLit : tone > 0.34 ? P.cobble : P.cobbleDark;
          // Body, then a rounded look by clipping the corners with mortar.
          dot(g, x, y, w, h, body);
          dot(g, x, y, 1, 1, P.mortar);
          dot(g, x + w - 1, y, 1, 1, P.mortar);
          dot(g, x, y + h - 1, 1, 1, P.mortar);
          dot(g, x + w - 1, y + h - 1, 1, 1, P.mortar);
          // Lit top edge and shadowed base.
          dot(g, x + 1, y, w - 2, 1, rowIndex % 2 ? P.cobbleHi : P.cobbleLit);
          dot(g, x + 1, y + h - 1, w - 2, 1, P.cobbleDark);
          // A speck of grit.
          if (random() > 0.6) dot(g, x + 2, y + 2, 1, 1, P.cobbleHi);
        }
      });

      // Grit and a stray pebble in the mortar.
      for (let i = 0; i < 8; i += 1) {
        dot(g, Math.floor(random() * size), Math.floor(random() * size), 1, 1, P.sandDark);
      }
    });
  }

  /** Packed dirt, used where a path meets grass. */
  function bakeDirt(variant) {
    return bakeTile((g, size) => {
      const random = seeded(variant * 331 + 7);
      for (let y = 0; y < size; y += 2) {
        for (let x = 0; x < size; x += 2) {
          const roll = random();
          dot(g, x, y, 2, 2, roll > 0.62 ? P.dirtLit : roll > 0.28 ? P.dirt : P.dirtDeep);
        }
      }
      for (let i = 0; i < 10; i += 1) {
        dot(g, Math.floor(random() * size), Math.floor(random() * size), 1, 1, P.sandDark);
      }
    });
  }

  /**
   * Water: horizontal ripple bands that step across four baked frames, so the
   * surface animates by swapping tiles rather than redrawing per pixel.
   */
  function bakeWater(frame) {
    return bakeTile((g, size) => {
      const random = seeded(frame * 211 + 5);

      for (let y = 0; y < size; y += 1) {
        // Bands of depth, shifted per frame so the whole surface drifts.
        const band = Math.sin((y + frame * 2) * 0.55);
        const base = band > 0.45 ? P.waterLit : band < -0.4 ? P.waterDeep : P.water;
        dot(g, 0, y, size, 1, base);
      }
      // Dither between the bands so the transitions are not hard lines.
      for (let y = 0; y < size; y += 2) {
        for (let x = (y / 2) % 2; x < size; x += 4) {
          dot(g, x, y, 2, 1, P.water);
        }
      }
      // Ripple highlights: short dashes that move with the frame.
      for (let i = 0; i < 5; i += 1) {
        const x = (Math.floor(random() * size) + frame * 3) % size;
        const y = Math.floor(random() * size);
        const w = 3 + Math.floor(random() * 4);
        dot(g, x, y, w, 1, P.waterLit);
        dot(g, x + 1, y - 1, Math.max(1, w - 2), 1, P.waterFoam);
      }
      // A couple of deep flecks for depth.
      for (let i = 0; i < 3; i += 1) {
        dot(g, Math.floor(random() * size), Math.floor(random() * size), 2, 1, P.waterDeep);
      }
    });
  }

  /** Bake everything once. Called from boot, before the first frame. */
  function bakeTextures() {
    for (let i = 0; i < 6; i += 1) TEX.grass.push(bakeGrass(i));
    for (let i = 0; i < 4; i += 1) TEX.path.push(bakePath(i));
    for (let i = 0; i < 3; i += 1) TEX.dirt.push(bakeDirt(i));
    for (let i = 0; i < 4; i += 1) TEX.water.push(bakeWater(i));
  }

  /* ======================================================================
     Map: 0 = grass, 1 = path, 2 = water, 3 = solid (building/prop)
     ====================================================================== */

  const tiles = new Uint8Array(COLS * ROWS);
  const tileAt = (cx, cy) => tiles[cy * COLS + cx];
  const setTile = (cx, cy, value) => {
    if (cx >= 0 && cy >= 0 && cx < COLS && cy < ROWS) tiles[cy * COLS + cx] = value;
  };

  /** Deterministic per-tile noise, so the grass texture never shimmers. */
  function hash(x, y) {
    const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
    return n - Math.floor(n);
  }

  /**
   * Buildings. `door` is the tile a villager walks to; `w`/`h` are tile spans.
   * Positions are hand-placed so the paths between them read as a village
   * rather than a grid.
   */
  const BUILDINGS = [
    { id: 'town_hall', name: 'Town Hall', x: 17, y: 3, w: 6, h: 4, roof: 'roofRed', door: { x: 20, y: 7 }, icon: '👑' },
    { id: 'watchtower', name: 'Watchtower', x: 3, y: 2, w: 3, h: 5, roof: 'roofBlue', door: { x: 4, y: 7 }, icon: '🔭', tower: true },
    { id: 'forge', name: 'The Forge', x: 31, y: 5, w: 5, h: 4, roof: 'roofRedDark', door: { x: 33, y: 9 }, icon: '🎨', forge: true },
    { id: 'scribe_cottage', name: "Scribe's Cottage", x: 4, y: 12, w: 5, h: 4, roof: 'roofPurple', door: { x: 6, y: 16 }, icon: '📜' },
    { id: 'gatehouse', name: 'Gatehouse', x: 17, y: 19, w: 5, h: 4, roof: 'stone', door: { x: 19, y: 18 }, icon: '🛡️', stone: true },
    { id: 'market', name: 'Oakhaven Market', x: 30, y: 13, w: 6, h: 4, roof: 'roofGreen', door: { x: 32, y: 17 }, icon: '🏪' },
    { id: 'bell_tower', name: 'Bell Tower', x: 25, y: 9, w: 3, h: 5, roof: 'thatch', door: { x: 26, y: 14 }, icon: '🔔', tower: true },
    { id: 'tavern', name: 'The Gilded Stag', x: 10, y: 6, w: 5, h: 4, roof: 'thatch', door: { x: 12, y: 10 }, icon: '🍺' },
    // The Bard's Theater: the tallest roof in the village, on the southern road.
    { id: 'theater', name: "The Bard's Theater", x: 27, y: 19, w: 6, h: 5, roof: 'tilePlum', door: { x: 30, y: 18 }, icon: '🎭', theater: true },
    // Vesper's observatory, in the gap between the cottage and the gatehouse.
    { id: 'observatory', name: 'The Observatory', x: 11, y: 12, w: 5, h: 4, roof: 'roofBlue', door: { x: 13, y: 16 }, icon: '🌙', tower: true },
    // The counting house, on the quiet south-western corner.
    { id: 'counting_house', name: 'The Counting House', x: 3, y: 18, w: 5, h: 4, roof: 'stone', door: { x: 5, y: 22 }, icon: '👑', stone: true },
  ];

  const buildingById = new Map(BUILDINGS.map((b) => [b.id, b]));

  /** Places a villager wanders to when idle. */
  const SPOTS = {
    tavern: { x: 12, y: 11 },
    picnic_north: { x: 14, y: 14 },
    picnic_south: { x: 22, y: 15 },
    well: { x: 19, y: 12 },
    orchard: { x: 34, y: 20 },
    dock: { x: 11, y: 20 },
    square: { x: 20, y: 10 },
    stage_door: { x: 33, y: 18 },
  };

  /** Decorative props: trees, the well, picnic tables, crates, the pond. */
  const PROPS = [];

  function buildMap() {
    // Base grass with a few darker patches.
    for (let y = 0; y < ROWS; y += 1) {
      for (let x = 0; x < COLS; x += 1) {
        setTile(x, y, 0);
      }
    }

    // Pond in the south-west.
    for (let y = 19; y < 23; y += 1) {
      for (let x = 4; x < 11; x += 1) {
        // Rounded edges so it is not a rectangle of water.
        const dx = (x - 7.5) / 3.6;
        const dy = (y - 21) / 2.0;
        if (dx * dx + dy * dy <= 1) setTile(x, y, 2);
      }
    }

    // Main horizontal road, then spurs to each door.
    for (let x = 1; x < COLS - 1; x += 1) setTile(x, 11, 1);
    for (let y = 7; y < 19; y += 1) setTile(20, y, 1);
    for (let y = 11; y < 18; y += 1) setTile(32, y, 1);
    for (let y = 7; y < 11; y += 1) setTile(4, y, 1);
    for (let y = 11; y < 17; y += 1) setTile(6, y, 1);
    for (let y = 9; y < 12; y += 1) setTile(33, y, 1);
    for (let y = 10; y < 15; y += 1) setTile(26, y, 1);
    for (let y = 10; y < 12; y += 1) setTile(12, y, 1);
    for (let x = 4; x <= 6; x += 1) setTile(x, 10, 1);
    for (let x = 26; x <= 33; x += 1) setTile(x, 10, 1);
    for (let x = 12; x <= 20; x += 1) setTile(x, 10, 1);
    // Southern road past the gatehouse, and the spur that reaches it.
    for (let x = 10; x <= 32; x += 1) setTile(x, 17, 1);
    for (let y = 11; y <= 17; y += 1) setTile(12, y, 1);
    for (let y = 17; y <= 21; y += 1) setTile(24, y, 1);
    for (let x = 12; x <= 24; x += 1) setTile(x, 21, 1);

    // Buildings become solid, and their door tile becomes path.
    BUILDINGS.forEach((b) => {
      for (let y = b.y; y < b.y + b.h; y += 1) {
        for (let x = b.x; x < b.x + b.w; x += 1) setTile(x, y, 3);
      }
      setTile(b.door.x, b.door.y, 1);
    });

    // Trees around the rim, avoiding anything already placed.
    const treeSpots = [
      [1, 4], [1, 8], [2, 15], [1, 17], [2, 23], [9, 2], [14, 2], [26, 2],
      [37, 3], [38, 7], [37, 11], [38, 15], [36, 19], [33, 21], [29, 21],
      [24, 20], [14, 18], [11, 17], [8, 16], [16, 6], [23, 16], [28, 18],
      [12, 22], [17, 23], [22, 22], [30, 8], [38, 22], [2, 11],
    ];
    treeSpots.forEach(([x, y]) => {
      if (tileAt(x, y) === 0) {
        setTile(x, y, 3);
        PROPS.push({ kind: 'tree', x, y, seed: hash(x, y) });
      }
    });

    // The well at the crossroads, picnic tables, market crates.
    PROPS.push({ kind: 'well', x: 19, y: 13 });
    setTile(19, 13, 3);
    PROPS.push({ kind: 'picnic', x: 14, y: 15 });
    setTile(14, 15, 3);
    PROPS.push({ kind: 'picnic', x: 22, y: 14 });
    setTile(22, 14, 3);
    PROPS.push({ kind: 'crates', x: 29, y: 18 });
    setTile(29, 18, 3);
    PROPS.push({ kind: 'crates', x: 35, y: 17 });
    setTile(35, 17, 3);
    PROPS.push({ kind: 'stump', x: 9, y: 14 });
    setTile(9, 14, 3);
    // Street lamps light the road but stand on the verge beside it, never in
    // the middle of it. Each anchor snaps to the first free grass tile next to
    // the road it is meant to light.
    const onVerge = (x, y) => {
      const candidates = [[x, y - 1], [x, y + 1], [x - 1, y], [x + 1, y], [x, y]];
      for (const [cx, cy] of candidates) {
        if (cx < 1 || cy < 1 || cx >= COLS - 1 || cy >= ROWS - 1) continue;
        if (tileAt(cx, cy) === 0) return { x: cx, y: cy };
      }
      return null;
    };
    [[18, 10], [24, 12], [15, 17], [30, 11], [21, 21]].forEach(([x, y]) => {
      const spot = onVerge(x, y);
      if (spot) PROPS.push({ kind: 'lamp', x: spot.x, y: spot.y, seed: hash(spot.x, spot.y) });
    });

    // The signpost belongs AT the crossroads, so it stays on the road tile.
    // It is decoration only and never blocks a route.
    PROPS.push({ kind: 'sign', x: 21, y: 11, seed: hash(21, 11) });

    // Bushes and barrels soften the gaps between buildings and the road.
    const solidProps = [
      ['bush', 16, 8], ['bush', 24, 6], ['bush', 8, 4], ['bush', 29, 3],
      ['bush', 35, 11], ['bush', 27, 16], ['bush', 10, 19], ['bush', 3, 18],
      ['barrel', 30, 12], ['barrel', 34, 12], ['barrel', 15, 5],
      ['barrel', 23, 18], ['barrel', 9, 5],
      ['cart', 28, 12],
    ];
    solidProps.forEach(([kind, x, y]) => {
      if (tileAt(x, y) !== 0) return;
      setTile(x, y, 3);
      PROPS.push({ kind, x, y, seed: hash(x, y) });
    });

    // A fence line along the orchard, and flower beds on open grass. Flowers
    // are decoration only — they never block a route.
    for (let x = 33; x <= 37; x += 1) {
      if (tileAt(x, 22) === 0) PROPS.push({ kind: 'fence', x, y: 22, seed: hash(x, 22) });
    }
    [[13, 12], [27, 12], [7, 9], [22, 8], [31, 20], [16, 22]].forEach(([x, y]) => {
      if (tileAt(x, y) === 0) PROPS.push({ kind: 'flowers', x, y, seed: hash(x, y) });
    });
  }

  const walkable = (cx, cy) => {
    if (cx < 0 || cy < 0 || cx >= COLS || cy >= ROWS) return false;
    const t = tileAt(cx, cy);
    return t === 0 || t === 1;
  };

  /* ======================================================================
     Pathfinding — A* over the tile grid, 4-directional
     ====================================================================== */

  function nearestWalkable(cx, cy) {
    if (walkable(cx, cy)) return { x: cx, y: cy };
    for (let radius = 1; radius <= 8; radius += 1) {
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          if (Math.abs(dx) !== radius && Math.abs(dy) !== radius) continue;
          if (walkable(cx + dx, cy + dy)) return { x: cx + dx, y: cy + dy };
        }
      }
    }
    return { x: 20, y: 11 };
  }

  /**
   * @returns {{x:number,y:number}[]} tile waypoints, excluding the start.
   */
  function findPath(from, to) {
    const start = nearestWalkable(from.x, from.y);
    const goal = nearestWalkable(to.x, to.y);
    if (start.x === goal.x && start.y === goal.y) return [];

    const key = (x, y) => y * COLS + x;
    const open = [{ x: start.x, y: start.y, g: 0, f: 0 }];
    const cameFrom = new Map();
    const best = new Map([[key(start.x, start.y), 0]]);
    const heuristic = (x, y) => Math.abs(x - goal.x) + Math.abs(y - goal.y);
    let found = false;
    let guard = 0;

    while (open.length && guard++ < 8000) {
      // Linear scan: the grid is 1000 cells, a heap is not worth the code.
      let bestIndex = 0;
      for (let i = 1; i < open.length; i += 1) {
        if (open[i].f < open[bestIndex].f) bestIndex = i;
      }
      const current = open.splice(bestIndex, 1)[0];

      if (current.x === goal.x && current.y === goal.y) {
        found = true;
        break;
      }

      const neighbours = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ];
      for (const [dx, dy] of neighbours) {
        const nx = current.x + dx;
        const ny = current.y + dy;
        if (!walkable(nx, ny)) continue;

        // Roads are cheaper, so villagers prefer them and traffic looks routed.
        const cost = current.g + (tileAt(nx, ny) === 1 ? 1 : 1.7);
        const k = key(nx, ny);
        if (best.has(k) && best.get(k) <= cost) continue;

        best.set(k, cost);
        cameFrom.set(k, current);
        open.push({ x: nx, y: ny, g: cost, f: cost + heuristic(nx, ny) });
      }
    }

    if (!found) return [];

    const cells = [];
    let cursor = { x: goal.x, y: goal.y };
    while (cursor) {
      cells.unshift(cursor);
      cursor = cameFrom.get(key(cursor.x, cursor.y));
    }
    return cells.slice(1);
  }

  /* ======================================================================
     Villager sprites
     ====================================================================== */

  const px = (x, y, w, h, colour) => {
    ctx.fillStyle = colour;
    ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
  };

  /* ======================================================================
     Villager sprites

     A ~30px tall figure anchored at the feet — roughly 1.25 tiles, which is
     the proportion 16-bit JRPGs use so a character reads clearly against
     24px scenery without towering over the doorways.

     The walk is the classic four-frame cycle: contact, down, contact, up.
     Legs swing opposite each other, arms counter-swing against the legs, and
     the whole body drops a pixel on the down frames. That vertical bob is the
     thing that separates a walk from a slide.
     ====================================================================== */

  /** Per frame: [legSwing, bodyDrop, armSwing]. */
  const WALK_CYCLE = [
    [3, 0, -2],
    [0, 1, 0],
    [-3, 0, 2],
    [0, 1, 0],
  ];

  /**
   * Draw one villager into any 2D context.
   *
   * @param {CanvasRenderingContext2D} target
   * @param {Object} villager palette + look fields
   * @param {Object} options {x, y (feet), scale, frame, facing, pose}
   */
  function drawVillagerSprite(target, villager, options) {
    const o = options || {};
    const originX = o.x || 0;
    const feetY = o.y || 0;
    const scale = o.scale || 1;
    const frame = (o.frame || 0) % 4;
    const facing = o.facing || 1;
    const pose = o.pose || 'idle';

    const p = (dx, dy, w, h, colour) => {
      target.fillStyle = colour;
      target.fillRect(
        Math.round(originX + dx * scale),
        Math.round(feetY + dy * scale),
        Math.max(1, Math.round(w * scale)),
        Math.max(1, Math.round(h * scale)),
      );
    };

    const walking = pose === 'walk';
    const [legSwing, bodyDrop, armSwing] = walking
      ? WALK_CYCLE[frame]
      : [0, pose === 'idle' && frame % 4 === 2 ? 1 : 0, 0];

    // Palette for this villager, with derived shades.
    const cloth = villager.color || '#7a5230';
    const clothLit = shade(cloth, 0.22);
    const clothDark = shade(cloth, -0.3);
    const clothDeep = shade(cloth, -0.52);
    const hair = villager.hair || P.outline;
    const hairLit = shade(hair, 0.28);
    const hairDeep = shade(hair, -0.35);
    const skin = villager.skin || P.skin;
    const skinLit = shade(skin, 0.2);
    const skinDark = shade(skin, -0.28);
    const boot = shade(villager.boot || '#4a3222', 0);
    const bootLit = shade(boot, 0.24);

    // Everything below is measured UP from the feet, so y is negative.
    const drop = bodyDrop;

    // ---- contact shadow, squashed and offset --------------------------------
    p(-7, -2, 14, 3, P.shadow);
    p(-5, -1, 10, 2, 'rgba(40,26,14,0.22)');

    // ---- legs and boots -----------------------------------------------------
    const frontLeg = facing > 0 ? legSwing : -legSwing;
    const backLeg = -frontLeg;

    // Back leg first so the front one overlaps it.
    p(-4 + backLeg * 0.5, -9 + drop, 4, 7, clothDeep);
    p(-4 + backLeg * 0.5, -3 + drop, 5, 3, shade(boot, -0.2));

    p(0 + frontLeg * 0.5, -9 + drop, 4, 7, clothDark);
    p(1 + frontLeg * 0.5, -9 + drop, 2, 6, clothLit);
    p(0 + frontLeg * 0.5, -3 + drop, 5, 3, boot);
    p(0 + frontLeg * 0.5, -3 + drop, 5, 1, bootLit);

    // ---- torso --------------------------------------------------------------
    const torsoTop = -22 + drop;
    p(-6, torsoTop, 12, 14, P.outline);
    p(-5, torsoTop + 1, 10, 12, cloth);
    // Lit side toward the facing direction, shadow behind.
    p(facing > 0 ? -5 : 1, torsoTop + 1, 4, 10, clothLit);
    p(facing > 0 ? 3 : -5, torsoTop + 1, 2, 12, clothDark);
    // Cloth folds.
    p(-3, torsoTop + 7, 6, 1, clothDark);
    p(-2, torsoTop + 10, 4, 1, clothDeep);
    // Belt with a buckle.
    p(-6, torsoTop + 11, 12, 3, shade(cloth, -0.62));
    p(-1, torsoTop + 11, 3, 3, P.gold);
    p(-1, torsoTop + 11, 3, 1, P.goldHi);
    // Collar.
    p(-3, torsoTop, 6, 2, clothDeep);

    // ---- arms ---------------------------------------------------------------
    const armTop = torsoTop + 3;
    const frontArm = facing > 0 ? armSwing : -armSwing;
    let workArm = 0;
    if (pose === 'hammer') workArm = frame % 2 === 0 ? -6 : 2;
    else if (pose === 'chop') workArm = frame % 2 === 0 ? -7 : 3;
    else if (pose === 'bell') workArm = frame % 2 === 0 ? -4 : 3;
    else if (pose === 'quill') workArm = frame % 2 === 0 ? 0 : 1;
    else if (pose === 'spyglass') workArm = -5;
    else if (pose === 'inspect') workArm = -3;
    else if (pose === 'trade') workArm = -1;

    // Back arm.
    p(facing > 0 ? -8 : 5, armTop - frontArm * 0.4, 3, 8, clothDark);
    p(facing > 0 ? -8 : 5, armTop + 8 - frontArm * 0.4, 3, 3, skinDark);

    // Front arm, raised by the work pose.
    const armY = armTop + workArm + frontArm * 0.4;
    p(facing > 0 ? 5 : -8, armY, 3, 8, cloth);
    p(facing > 0 ? 5 : -8, armY, 3, 6, clothLit);
    p(facing > 0 ? 5 : -8, armY + 8, 3, 3, skin);
    p(facing > 0 ? 5 : -8, armY + 10, 3, 1, skinDark);

    // ---- head ---------------------------------------------------------------
    const headTop = torsoTop - 12;
    p(-6, headTop, 12, 13, P.outline);
    p(-5, headTop + 1, 10, 11, skin);
    // Lit cheek and shadowed jaw.
    p(facing > 0 ? -1 : -5, headTop + 2, 6, 7, skinLit);
    p(-5, headTop + 9, 10, 2, skinDark);
    p(-4, headTop + 11, 8, 1, shade(skin, -0.45));
    // Ear on the trailing side.
    p(facing > 0 ? -6 : 5, headTop + 5, 2, 3, skin);
    p(facing > 0 ? -6 : 5, headTop + 6, 1, 2, skinDark);

    // ---- face ---------------------------------------------------------------
    const eyeX = facing > 0 ? -1 : -3;
    // Brow, then a two-pixel eye with a white and a pupil.
    p(eyeX - 1, headTop + 4, 3, 1, hairDeep);
    p(eyeX + 3, headTop + 4, 3, 1, hairDeep);
    p(eyeX, headTop + 5, 2, 2, P.white);
    p(eyeX + 4, headTop + 5, 2, 2, P.white);
    p(eyeX, headTop + 5, 1, 2, villager.eye || '#3a2a1c');
    p(eyeX + 4, headTop + 5, 1, 2, villager.eye || '#3a2a1c');
    // Nose and mouth.
    p(facing > 0 ? eyeX + 5 : eyeX - 1, headTop + 7, 1, 1, skinDark);
    p(eyeX + 1, headTop + 9, 3, 1, shade(skin, -0.5));

    // ---- hair ---------------------------------------------------------------
    // Mass, then a lit band, then strands breaking the silhouette.
    p(-6, headTop - 2, 12, 6, hair);
    p(-7, headTop + 1, 3, 6, hair);
    p(4, headTop + 1, 3, 6, hair);
    p(-4, headTop - 3, 8, 2, hair);
    p(-3, headTop - 2, 6, 2, hairLit);
    p(facing > 0 ? 1 : -4, headTop - 1, 3, 1, hairLit);
    // Fringe over the brow.
    p(facing > 0 ? -1 : -5, headTop + 2, 6, 2, hair);
    p(facing > 0 ? -6 : 1, headTop + 2, 5, 2, hairDeep);
    // Stray tufts.
    p(-5, headTop - 4, 2, 2, hair);
    p(2, headTop - 4, 2, 2, hair);

    // ---- headgear -----------------------------------------------------------
    drawHeadgear(p, villager, headTop, facing, cloth, clothLit, clothDark);

    // ---- held prop ----------------------------------------------------------
    drawHeldProp(p, pose, facing, armY, villager);
  }

  /** Hats, helms, hoods and crowns — the fastest way to tell villagers apart. */
  function drawHeadgear(p, villager, headTop, facing, cloth, clothLit, clothDark) {
    switch (villager.hat) {
      case 'crown':
        p(-6, headTop - 6, 12, 4, P.gold);
        p(-6, headTop - 6, 12, 1, P.goldHi);
        p(-5, headTop - 9, 2, 3, P.gold);
        p(-1, headTop - 11, 2, 5, P.gold);
        p(3, headTop - 9, 2, 3, P.gold);
        p(-1, headTop - 11, 1, 1, P.goldHi);
        p(-1, headTop - 5, 2, 2, '#c8434f');
        break;

      case 'helm':
        p(-7, headTop - 4, 14, 7, P.stoneDark);
        p(-7, headTop - 4, 14, 2, P.stoneHi);
        p(-7, headTop + 3, 14, 2, P.stoneDeep);
        // Nasal bar and a plume.
        p(facing > 0 ? 1 : -2, headTop + 2, 2, 6, P.stone);
        p(-1, headTop - 9, 3, 6, '#c8434f');
        p(-1, headTop - 9, 1, 5, '#e05a66');
        break;

      case 'hood':
        p(-8, headTop - 4, 16, 8, clothDark);
        p(-7, headTop - 3, 14, 5, cloth);
        p(-6, headTop - 3, 6, 2, clothLit);
        // Sides falling past the jaw, and a shoulder drape.
        p(-8, headTop + 3, 3, 9, clothDark);
        p(5, headTop + 3, 3, 9, clothDark);
        p(-9, headTop + 10, 18, 4, clothDeepOf(cloth));
        break;

      case 'cap':
        p(-6, headTop - 4, 12, 5, clothDark);
        p(-5, headTop - 4, 10, 2, clothLit);
        // Brim, pointing the way they face.
        p(facing > 0 ? 4 : -10, headTop - 1, 6, 2, clothDark);
        p(facing > 0 ? 4 : -10, headTop - 1, 6, 1, cloth);
        // Feather.
        p(facing > 0 ? -7 : 5, headTop - 8, 2, 6, '#c8434f');
        p(facing > 0 ? -7 : 5, headTop - 9, 2, 2, '#e05a66');
        break;

      case 'kerchief':
        p(-6, headTop - 3, 12, 4, '#c8434f');
        p(-6, headTop - 3, 12, 1, '#e05a66');
        p(-5, headTop - 2, 4, 1, '#f0e08a');
        p(2, headTop - 1, 2, 1, '#f0e08a');
        // Knot trailing behind.
        p(facing > 0 ? -8 : 6, headTop - 1, 3, 3, '#c8434f');
        p(facing > 0 ? -10 : 8, headTop, 2, 4, '#a8353f');
        break;

      case 'feathered':
        // A slouched player's cap with a long plume — theatrical on purpose.
        p(-7, headTop - 5, 14, 6, clothDark);
        p(-6, headTop - 5, 12, 2, clothLit);
        p(facing > 0 ? 4 : -10, headTop - 2, 6, 2, clothDark);
        p(facing > 0 ? 4 : -10, headTop - 2, 6, 1, cloth);
        // Plume, arcing back and up.
        p(facing > 0 ? -9 : 6, headTop - 8, 3, 4, '#e0a53f');
        p(facing > 0 ? -11 : 8, headTop - 12, 3, 5, '#f7cf76');
        p(facing > 0 ? -12 : 9, headTop - 15, 2, 4, '#e0a53f');
        break;

      case 'bandana':
        p(-6, headTop - 2, 12, 3, '#4a6d8a');
        p(-6, headTop - 2, 12, 1, '#6a8daa');
        p(facing > 0 ? -8 : 6, headTop, 3, 5, '#4a6d8a');
        break;

      default:
        break;
    }
  }

  /** Shorthand used by the hood, which needs a fourth step of its ramp. */
  function clothDeepOf(cloth) {
    return shade(cloth, -0.52);
  }

  /** Tools and papers, matched to the work animation. */
  function drawHeldProp(p, pose, facing, armY, villager) {
    const side = facing > 0 ? 1 : -1;
    const hx = facing > 0 ? 7 : -11;

    switch (pose) {
      case 'hammer':
        p(hx, armY + 6, 2, 8, P.wood);
        p(hx, armY + 6, 1, 8, P.woodHi);
        p(hx - 2 * side + (facing > 0 ? 0 : -2), armY + 1, 7, 5, P.stoneDark);
        p(hx - 2 * side + (facing > 0 ? 0 : -2), armY + 1, 7, 2, P.stoneLit);
        break;

      case 'chop':
        p(hx, armY + 5, 2, 9, P.wood);
        p(hx - 2, armY + 1, 6, 4, P.stone);
        p(hx - 2, armY + 1, 6, 1, P.stoneHi);
        break;

      case 'quill':
        // Feather quill with a nib, plus an inkpot on the desk side.
        p(hx, armY + 4, 1, 5, P.white);
        p(hx, armY, 2, 5, '#f2e6c8');
        p(hx + 1, armY - 2, 1, 3, '#d8ccae');
        p(hx - 4 * side, armY + 9, 4, 4, '#2a2a3a');
        p(hx - 4 * side, armY + 9, 4, 1, '#454055');
        break;

      case 'spyglass':
        p(hx - 1, armY + 5, 8, 3, P.stoneDeep);
        p(hx - 1, armY + 5, 8, 1, P.stoneLit);
        p(hx + 7, armY + 5, 2, 3, P.gold);
        p(hx + 8, armY + 6, 1, 1, '#a8d4ff');
        break;

      // Marlow's ledger: a small open book he tallies prices in. The roster
      // named this animation from the start but nothing drew it, so the deal
      // Scout worked with empty hands.
      // Vesper's star chart: a rolled sheet with plotted points, read by lamp.
      case 'starchart':
        p(hx - 4, armY + 3, 13, 9, P.stoneDeep);
        p(hx - 3, armY + 4, 11, 7, '#1d2440');
        for (let i = 0; i < 5; i += 1) {
          const sxp = hx - 2 + ((i * 7) % 10);
          const syp = armY + 5 + ((i * 3) % 5);
          p(sxp, syp, 1, 1, i % 2 ? P.gold : '#a8d4ff');
        }
        p(hx - 4, armY + 3, 13, 1, P.stoneLit);
        break;

      // Aldric's abacus: beads on wires, the council's least glamorous tool.
      case 'abacus':
        p(hx - 3, armY + 3, 11, 9, P.woodDeep);
        p(hx - 2, armY + 4, 9, 7, P.woodDark);
        for (let r = 0; r < 3; r += 1) {
          const ry = armY + 5 + r * 2;
          p(hx - 2, ry, 9, 1, P.woodHi);
          p(hx - 1 + r, ry - 1, 2, 2, P.gold);
          p(hx + 4 - r, ry - 1, 2, 2, P.stoneLit);
        }
        break;

      case 'ledger':
        p(hx - 3, armY + 4, 11, 8, P.woodDeep);
        p(hx - 2, armY + 5, 9, 6, '#efe3c8');
        p(hx + 2, armY + 4, 1, 8, P.woodDark);
        for (let i = 0; i < 3; i += 1) {
          p(hx - 1, armY + 6 + i * 2, 3, 1, P.stoneDeep);
          p(hx + 4, armY + 6 + i * 2, 2, 1, P.stoneDeep);
        }
        p(hx + 6, armY + 2, 1, 4, P.gold);
        break;

      case 'scroll':
      case 'ponder':
        p(hx - 1, armY + 4, 7, 9, P.white);
        p(hx - 1, armY + 4, 7, 1, '#d8ccae');
        p(hx, armY + 6, 5, 1, '#a89878');
        p(hx, armY + 8, 3, 1, '#a89878');
        p(hx - 2, armY + 4, 2, 9, '#d8ccae');
        break;

      case 'trade':
        // A crate of goods held at the hip.
        p(hx - 1, armY + 4, 9, 8, P.woodDeep);
        p(hx, armY + 5, 7, 6, P.wood);
        p(hx, armY + 5, 7, 1, P.woodHi);
        p(hx + 1, armY + 3, 3, 3, '#c8434f');
        p(hx + 4, armY + 3, 3, 3, '#6d9a3c');
        break;

      case 'bell':
        p(hx, armY + 3, 2, 3, P.woodDark);
        p(hx - 2, armY + 5, 6, 6, P.gold);
        p(hx - 2, armY + 5, 6, 2, P.goldHi);
        p(hx - 3, armY + 10, 8, 2, P.gold);
        p(hx, armY + 12, 2, 2, shade(P.gold, -0.4));
        break;

      case 'lute':
        // Pear-shaped body, angled neck, and strings he is actually playing.
        p(hx - 3, armY + 4, 10, 10, '#7a4a24');
        p(hx - 2, armY + 5, 8, 8, '#a06d38');
        p(hx - 1, armY + 6, 4, 3, '#5a3418');
        p(hx + 5, armY, 3, 8, '#5a3418');
        p(hx + 6, armY - 3, 3, 4, '#7a4a24');
        for (let i = 0; i < 3; i += 1) {
          p(hx - 1 + i * 2, armY + 4, 1, 10, '#e0cba0');
        }
        break;

      case 'inspect':
        p(hx - 1, armY + 2, 8, 8, P.stoneDark);
        p(hx, armY + 3, 6, 6, 'rgba(168,212,255,0.42)');
        p(hx + 1, armY + 4, 2, 2, 'rgba(255,255,255,0.6)');
        p(hx + 6, armY + 9, 4, 5, P.wood);
        break;

      default:
        break;
    }
  }

  /** Lighten (amount > 0) or darken a hex colour, staying in the same hue. */
  function shade(hex, amount) {
    if (typeof hex !== 'string' || hex[0] !== '#') return hex;
    const value = hex.slice(1);
    const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
    const num = parseInt(full, 16);
    let r = (num >> 16) & 255;
    let g = (num >> 8) & 255;
    let b = num & 255;
    // Warm the highlights and cool-shift the shadows only slightly, which is
    // what keeps a ramp from looking like a brightness slider.
    const t = amount < 0 ? 0 : 255;
    const strength = Math.abs(amount);
    r = Math.round((t - r) * strength * (amount > 0 ? 1.05 : 0.95) + r);
    g = Math.round((t - g) * strength + g);
    b = Math.round((t - b) * strength * (amount > 0 ? 0.9 : 1.05) + b);
    const clamp = (v) => Math.max(0, Math.min(255, v));
    return `rgb(${clamp(r)},${clamp(g)},${clamp(b)})`;
  }

  /* ======================================================================
     Villager entities
     ====================================================================== */

  /** Per-villager sprite flavour, keyed by the backend's agent id. */
  const LOOKS = {
    mayor: { hat: 'crown', hair: '#8a8378', skin: '#f4d3ac', eye: '#4a6d8a', boot: '#5a3a22' },
    scout: { hat: 'cap', hair: '#5b3a22', skin: '#e0b184', eye: '#3f6d31', boot: '#4a3a24' },
    crafter: { hat: 'kerchief', hair: '#2e211c', skin: '#c08d5e', eye: '#5e3a22', boot: '#3a2a1c' },
    scribe: { hat: 'hood', hair: '#a4291f', skin: '#f4d3ac', eye: '#5a4a7d', boot: '#4a3a52' },
    guard: { hat: 'helm', hair: '#241d18', skin: '#8e6039', eye: '#3a4a63', boot: '#3a3238' },
    merchant: { hat: 'bandana', hair: '#4a4034', skin: '#e0b184', eye: '#4a6b3a', boot: '#5a4222' },
    crier: { hat: 'cap', hair: '#3a2a20', skin: '#e0b184', eye: '#8a6a2f', boot: '#6b4a2c' },
    // Finneas: the only villager in a feathered cap and a player's motley.
    bard: { hat: 'feathered', hair: '#6b3a52', skin: '#f4d3ac', eye: '#7a4a8a', boot: '#4a2a3a' },
  };

  /** Idle activities a villager rolls between when no job is running. */
  const IDLE_POSES = ['idle', 'scroll', 'chop', 'idle', 'ponder'];

  class Villager {
    constructor(info) {
      Object.assign(this, info, LOOKS[info.id] || {});
      const home = buildingById.get(info.home);
      const start = home ? home.door : SPOTS.square;
      this.tx = start.x;
      this.ty = start.y;
      this.x = start.x * TILE + TILE / 2;
      this.y = start.y * TILE + TILE / 2;
      this.path = [];
      this.speed = 26; // pixels per second
      this.facing = 1;
      this.pose = 'idle';
      this.mode = 'idle'; // idle | travelling | working | lingering
      this.status = 'idle';
      this.task = info.idleLine || '';
      this.detail = '';
      this.lastOutput = null;
      this.progress = 0;
      this.nextRoll = 2 + Math.random() * 8;
      this.lingerFor = 0;
      this.frame = 0;
      this.frameClock = 0;
      this.emote = null;
      this.emoteFor = 0;
    }

    get tile() {
      return { x: Math.floor(this.x / TILE), y: Math.floor(this.y / TILE) };
    }

    /** Send this villager to a tile. */
    goTo(target, mode) {
      const path = findPath(this.tile, target);
      this.path = path;
      this.mode = path.length ? mode || 'travelling' : mode || 'working';
      if (!path.length) this.arrive();
    }

    /** Called when the last waypoint is reached. */
    arrive() {
      this.path = [];
      if (this.mode === 'travelling') {
        this.mode = this.status === 'working' ? 'working' : 'lingering';
        this.lingerFor = 3 + Math.random() * 6;
      } else if (this.mode !== 'working') {
        this.mode = 'lingering';
        this.lingerFor = 3 + Math.random() * 6;
      }
      this.pose = this.mode === 'working' ? this.workAnimation || 'idle' : this.idlePose();
    }

    idlePose() {
      return IDLE_POSES[Math.floor(Math.random() * IDLE_POSES.length)];
    }

    /** React to a state push from the backend. */
    setState(agent) {
      const wasWorking = this.status === 'working';
      this.status = agent.status;
      this.task = agent.task || this.idleLine;
      this.detail = agent.detail || '';
      this.progress = agent.progress || 0;
      if (agent.lastOutput) this.lastOutput = agent.lastOutput;

      if (agent.status === 'working') {
        if (!wasWorking) {
          const home = buildingById.get(this.home);
          this.goTo(home ? home.door : SPOTS.square, 'travelling');
          this.emitEmote('❗');
        }
        this.mode = this.path.length ? 'travelling' : 'working';
      } else if (agent.status === 'done') {
        this.emitEmote('✅');
        this.mode = 'lingering';
        this.lingerFor = 4;
        this.pose = 'idle';
      } else if (agent.status === 'error') {
        this.emitEmote('❌');
        this.mode = 'lingering';
        this.lingerFor = 4;
        this.pose = 'idle';
      }
    }

    emitEmote(symbol) {
      this.emote = symbol;
      this.emoteFor = 2.2;
    }

    update(dt) {
      // Animation clock: 8 frames/sec.
      this.frameClock += dt;
      if (this.frameClock > 0.125) {
        this.frameClock = 0;
        this.frame = (this.frame + 1) % 4;
      }
      if (this.emoteFor > 0) this.emoteFor -= dt;

      // Follow the path.
      if (this.path.length) {
        const next = this.path[0];
        const targetX = next.x * TILE + TILE / 2;
        const targetY = next.y * TILE + TILE / 2;
        const dx = targetX - this.x;
        const dy = targetY - this.y;
        const dist = Math.hypot(dx, dy);
        const step = this.speed * (this.status === 'working' ? 1.7 : 1) * dt;

        if (dist <= step) {
          this.x = targetX;
          this.y = targetY;
          this.path.shift();
          if (!this.path.length) this.arrive();
        } else {
          this.x += (dx / dist) * step;
          this.y += (dy / dist) * step;
          if (Math.abs(dx) > 0.5) this.facing = dx > 0 ? 1 : -1;
        }
        this.pose = 'walk';
        return;
      }

      // Working: hold the work animation at the building.
      if (this.mode === 'working' || this.status === 'working') {
        this.pose = this.workAnimation || 'idle';
        return;
      }

      // Idle: linger, then wander somewhere else.
      this.lingerFor -= dt;
      this.nextRoll -= dt;
      if (this.lingerFor <= 0 && this.nextRoll <= 0) {
        this.nextRoll = 6 + Math.random() * 14;
        const keys = Object.keys(SPOTS);
        const spot = SPOTS[keys[Math.floor(Math.random() * keys.length)]];
        // Half the time head home instead, so villagers stay recognisable.
        const home = buildingById.get(this.home);
        const target = Math.random() < 0.45 && home ? home.door : spot;
        this.goTo(target, 'travelling');
      }
    }
  }

  /* ======================================================================
     World rendering
     ====================================================================== */

  function drawGround() {
    const waterFrame = TEX.water[Math.floor(clock * 3) % TEX.water.length];

    for (let y = 0; y < ROWS; y += 1) {
      for (let x = 0; x < COLS; x += 1) {
        const t = tileAt(x, y);
        const sx = x * TILE;
        const sy = y * TILE;
        const pick = Math.floor(hash(x, y) * 1000);

        if (t === 2) {
          ctx.drawImage(waterFrame, sx, sy);
          continue;
        }

        // Grass sits under everything, so a path or building edge never gaps.
        ctx.drawImage(TEX.grass[pick % TEX.grass.length], sx, sy);

        if (t === 1) {
          ctx.drawImage(TEX.path[pick % TEX.path.length], sx, sy);
        }
      }
    }

    // Second pass: the transitions. Drawn after the whole base so an edge is
    // never painted over by the neighbour that comes later in the loop.
    for (let y = 0; y < ROWS; y += 1) {
      for (let x = 0; x < COLS; x += 1) {
        const t = tileAt(x, y);
        if (t === 1) drawPathEdges(x, y);
        else if (t === 2) drawShore(x, y);
      }
    }
  }

  /**
   * Where cobble meets grass, scatter loose stones and let grass creep over the
   * kerb. Hard-edged tiles look stamped; broken edges look walked on.
   */
  function drawPathEdges(cx, cy) {
    const sx = cx * TILE;
    const sy = cy * TILE;
    const random = seeded(cx * 73 + cy * 31 + 1);

    const sides = [
      { dx: 0, dy: -1, x: sx, y: sy, w: TILE, h: 3, horizontal: true },
      { dx: 0, dy: 1, x: sx, y: sy + TILE - 3, w: TILE, h: 3, horizontal: true },
      { dx: -1, dy: 0, x: sx, y: sy, w: 3, h: TILE, horizontal: false },
      { dx: 1, dy: 0, x: sx + TILE - 3, y: sy, w: 3, h: TILE, horizontal: false },
    ];

    sides.forEach((side) => {
      if (tileAt(cx + side.dx, cy + side.dy) === 1) return;

      // Grass fringe creeping over the edge, in irregular runs.
      const steps = side.horizontal ? TILE : TILE;
      for (let i = 0; i < steps; i += 2) {
        if (random() > 0.55) continue;
        const depth = 1 + Math.floor(random() * 2);
        if (side.horizontal) {
          px(side.x + i, side.dy < 0 ? sy : sy + TILE - depth, 2, depth, P.grassDark);
          px(side.x + i, side.dy < 0 ? sy : sy + TILE - 1, 2, 1, P.grassLit);
        } else {
          px(side.dx < 0 ? sx : sx + TILE - depth, side.y + i, depth, 2, P.grassDark);
          px(side.dx < 0 ? sx : sx + TILE - 1, side.y + i, 1, 2, P.grassLit);
        }
      }
      // A loose kerb stone or two spilling onto the grass.
      if (random() > 0.45) {
        const ox = side.horizontal ? Math.floor(random() * (TILE - 5)) : side.dx < 0 ? -3 : TILE + 1;
        const oy = side.horizontal ? (side.dy < 0 ? -3 : TILE + 1) : Math.floor(random() * (TILE - 5));
        px(sx + ox, sy + oy, 3, 2, P.cobbleDark);
        px(sx + ox, sy + oy, 3, 1, P.cobble);
      }
    });
  }

  /** Sand, foam and a wet line wherever water meets land. */
  function drawShore(cx, cy) {
    const sx = cx * TILE;
    const sy = cy * TILE;
    const foamPhase = Math.sin(clock * 1.6 + cx * 0.7 + cy * 0.4);

    const edges = [
      { dx: 0, dy: -1, x: sx, y: sy, w: TILE, h: 4, horizontal: true, near: true },
      { dx: 0, dy: 1, x: sx, y: sy + TILE - 4, w: TILE, h: 4, horizontal: true, near: false },
      { dx: -1, dy: 0, x: sx, y: sy, w: 4, h: TILE, horizontal: false },
      { dx: 1, dy: 0, x: sx + TILE - 4, y: sy, w: 4, h: TILE, horizontal: false },
    ];

    edges.forEach((edge) => {
      if (tileAt(cx + edge.dx, cy + edge.dy) === 2) return;

      // Wet sand band on the water side of the boundary.
      px(edge.x, edge.y, edge.w, edge.h, P.sandDark);
      if (edge.horizontal) {
        px(edge.x, edge.dy < 0 ? edge.y : edge.y + 2, edge.w, 2, P.sand);
      } else {
        px(edge.dx < 0 ? edge.x : edge.x + 2, edge.y, 2, edge.h, P.sand);
      }

      // Foam line that breathes in and out.
      if (foamPhase > 0) {
        if (edge.horizontal) {
          const fy = edge.dy < 0 ? edge.y + edge.h : edge.y - 1;
          for (let i = 0; i < TILE; i += 3) {
            if ((i + cx * 2) % 6 < 3) px(edge.x + i, fy, 2, 1, P.waterFoam);
          }
        } else {
          const fx = edge.dx < 0 ? edge.x + edge.w : edge.x - 1;
          for (let i = 0; i < TILE; i += 3) {
            if ((i + cy * 2) % 6 < 3) px(fx, edge.y + i, 1, 2, P.waterFoam);
          }
        }
      }

      // Reeds along the northern bank.
      if (edge.dy < 0 && (cx + cy) % 3 === 0) {
        const rx = sx + 4 + ((cx * 7) % 12);
        px(rx, sy - 5, 1, 6, P.grassDeep);
        px(rx + 2, sy - 7, 1, 8, P.grassDark);
        px(rx + 4, sy - 4, 1, 5, P.grassDeep);
        px(rx + 2, sy - 8, 1, 2, P.grassDry);
      }
    });
  }

  /* ======================================================================
     Buildings — built from parts, not boxes.

     Each cottage is a stone footing, a timber-framed plaster body with real
     posts and braces, a roof laid shingle by shingle in offset courses, then
     the fittings: shuttered windows, a planked door with a frame and handle,
     a window box, a chimney with courses, and a hanging sign.
     ====================================================================== */

  /** Roof ramps, keyed by the building's `roof` field. */
  const ROOFS = {
    slate: [P.slateDeep, P.slateDark, P.slate, P.slateLit],
    tileRed: [P.tileRedDeep, P.tileRedDark, P.tileRed, P.tileRedLit],
    tileGreen: [P.tileGreenDark, P.tileGreenDark, P.tileGreen, P.tileGreenLit],
    tilePlum: [P.tilePlumDark, P.tilePlumDark, P.tilePlum, P.tilePlumLit],
    thatch: [P.thatchDark, P.thatchDark, P.thatch, P.thatchLit],
  };

  function drawBuilding(b) {
    const sx = b.x * TILE;
    const sy = b.y * TILE;
    const w = b.w * TILE;
    const h = b.h * TILE;
    const lit = activeBuildings.has(b.id);

    const roofH = Math.round(h * 0.46);
    const wallY = sy + roofH;
    const wallH = h - roofH;
    const footY = sy + h - 7;

    // ---- ground shadow, soft and offset ------------------------------------
    px(sx + 4, sy + h - 2, w - 8, 5, P.shadow);
    px(sx + 8, sy + h + 3, w - 16, 2, P.shadowSoft);

    // ---- stone footing -----------------------------------------------------
    px(sx - 1, footY, w + 2, 8, P.stoneDeep);
    const footRandom = seeded(b.x * 31 + b.y * 17);
    for (let i = 0; i < w; i += 7) {
      const roll = footRandom();
      px(sx + i, footY + 1, 6, 5, roll > 0.6 ? P.stoneLit : roll > 0.3 ? P.stone : P.stoneDark);
      px(sx + i, footY + 1, 6, 1, P.stoneHi);
    }
    px(sx - 1, footY + 6, w + 2, 2, P.stoneDeep);

    // ---- wall body ---------------------------------------------------------
    px(sx, wallY, w, wallH - 6, P.outline);
    if (b.stone) {
      // Coursed stone with staggered joints.
      const random = seeded(b.x * 77 + b.y * 13);
      for (let row = 0; row + 6 < wallH - 8; row += 6) {
        const offset = (row / 6) % 2 ? 5 : 0;
        for (let i = -offset; i < w - 2; i += 10) {
          const roll = random();
          const body = roll > 0.66 ? P.stoneLit : roll > 0.33 ? P.stone : P.stoneDark;
          px(sx + 2 + Math.max(0, i), wallY + 2 + row, Math.min(9, w - 4 - i), 5, body);
          px(sx + 2 + Math.max(0, i), wallY + 2 + row, Math.min(9, w - 4 - i), 1, P.stoneHi);
        }
      }
    } else {
      // Plaster panel, dithered rather than flat.
      const random = seeded(b.x * 53 + b.y * 29);
      for (let yy = 1; yy < wallH - 8; yy += 2) {
        for (let xx = 2; xx < w - 2; xx += 2) {
          const roll = random();
          px(sx + xx, wallY + yy, 2, 2, roll > 0.7 ? P.plasterLit : roll > 0.22 ? P.plaster : P.plasterDark);
        }
      }
      // Timber frame: sill, head, corner posts, and a brace per bay.
      px(sx + 1, wallY, w - 2, 4, P.woodDark);
      px(sx + 1, wallY + 1, w - 2, 1, P.woodLit);
      px(sx + 1, wallY + wallH - 11, w - 2, 4, P.woodDark);
      px(sx + 1, wallY + wallH - 10, w - 2, 1, P.wood);
      px(sx + 1, wallY, 4, wallH - 7, P.woodDark);
      px(sx + w - 5, wallY, 4, wallH - 7, P.woodDark);
      px(sx + 2, wallY + 2, 1, wallH - 11, P.woodLit);

      for (let bay = 1; bay < b.w; bay += 1) {
        const bx = sx + bay * TILE - 2;
        px(bx, wallY + 3, 4, wallH - 13, P.woodDark);
        px(bx + 1, wallY + 4, 1, wallH - 15, P.wood);
        // Diagonal brace, stepped like real pixel art.
        for (let i = 0; i < 7; i += 1) {
          px(bx - 5 + i, wallY + 5 + i, 3, 2, P.woodDark);
        }
      }
    }

    // ---- roof: shingle courses --------------------------------------------
    const ramp = ROOFS[b.roof] || ROOFS.slate;
    const courses = Math.max(5, Math.round(roofH / 5));
    const overhang = 6;

    // Width of the topmost course, kept so the ridge cap can match it.
    let topW = 0;

    for (let c = 0; c < courses; c += 1) {
      const t = c / courses;
      // The inset SHRINKS as the courses descend, so the roof is narrow at the
      // ridge and widest at the eaves. It used to grow, which built the
      // triangle upside down: 204px across the ridge tapering to 48px at the
      // gutter on a six-tile building.
      const spread = (c + 1) / courses;
      const inset = Math.round((1 - spread) * (w / 2 - 8));
      const y = sy + Math.round(t * roofH);
      const rowW = w - inset * 2 + overhang * 2;
      const rowX = sx + inset - overhang;
      const rowH = Math.ceil(roofH / courses) + 1;
      if (c === 0) topW = rowW;

      // Lit along the ridge where the sky hits it, deepening toward the eave.
      const body = c === 0 ? ramp[3] : c < courses * 0.4 ? ramp[2] : ramp[1];
      px(rowX - 1, y, rowW + 2, rowH + 1, P.outline);
      px(rowX, y, rowW, rowH, body);

      if (b.roof === 'thatch') {
        // Thatch: vertical straw strokes rather than tiles.
        const random = seeded(b.x * 19 + c * 7);
        for (let i = 0; i < rowW; i += 2) {
          const len = 2 + Math.floor(random() * (rowH - 1));
          px(rowX + i, y + rowH - len, 1, len, random() > 0.5 ? ramp[3] : ramp[0]);
        }
        px(rowX, y + rowH - 1, rowW, 1, ramp[0]);
      } else {
        // Individual shingles, offset every other course.
        const shingle = 7;
        const offset = c % 2 ? Math.round(shingle / 2) : 0;
        for (let i = -offset; i < rowW; i += shingle) {
          const sxx = rowX + Math.max(0, i);
          const sww = Math.min(shingle - 1, rowW - Math.max(0, i));
          if (sww <= 0) continue;
          px(sxx, y + 1, sww, rowH - 2, body);
          // Lit top lip and the dark shadow the course above casts.
          px(sxx, y + 1, sww, 1, ramp[3]);
          px(sxx, y + rowH - 2, sww, 1, ramp[0]);
          px(sxx + sww, y + 1, 1, rowH - 2, ramp[0]);
        }
      }
    }

    // Ridge cap, sized to the topmost course rather than to the building: the
    // cap has to sit ON the ridge, and the ridge is now the narrow end.
    const ridgeW = Math.max(12, topW);
    px(sx + (w - ridgeW) / 2 - 4, sy - 3, ridgeW + 8, 5, P.outline);
    px(sx + (w - ridgeW) / 2 - 3, sy - 2, ridgeW + 6, 3, ramp[3]);
    px(sx - overhang - 1, wallY - 4, w + overhang * 2 + 2, 4, P.woodDark);
    px(sx - overhang - 1, wallY - 4, w + overhang * 2 + 2, 1, P.woodHi);

    // ---- door --------------------------------------------------------------
    // Sized so a ~40px villager reads as fitting through it, not ducking.
    const doorW = 20;
    const doorH = 34;
    const dx = Math.round(b.door.x * TILE + (TILE - doorW) / 2);
    const dy = footY - doorH + 2;

    px(dx - 3, dy - 3, doorW + 6, doorH + 4, P.woodDeep);
    px(dx - 2, dy - 2, doorW + 4, doorH + 2, P.wood);
    px(dx, dy, doorW, doorH, P.woodDeep);
    // Planks with visible joints.
    for (let i = 0; i < doorW; i += 5) {
      px(dx + i + 1, dy + 1, 4, doorH - 2, P.woodDark);
      px(dx + i + 1, dy + 1, 1, doorH - 2, P.wood);
    }
    // Iron straps, handle, and the lit gap beneath when someone is inside.
    px(dx, dy + 5, doorW, 2, P.stoneDark);
    px(dx, dy + doorH - 9, doorW, 2, P.stoneDark);
    px(dx + doorW - 5, dy + doorH / 2 - 2, 3, 4, P.gold);
    px(dx + doorW - 5, dy + doorH / 2 - 2, 3, 1, P.goldHi);
    // Arched head.
    px(dx + 2, dy - 4, doorW - 4, 2, P.woodDark);
    px(dx + 5, dy - 6, doorW - 10, 2, P.woodDark);
    if (lit) px(dx + 1, dy + doorH - 2, doorW - 2, 2, 'rgba(255,207,107,0.45)');

    // ---- windows -----------------------------------------------------------
    const winY = wallY + Math.round(wallH * 0.28);
    for (let i = 0; i < b.w; i += 1) {
      const wx = sx + i * TILE + 6;
      if (Math.abs(wx - dx) < 24) continue;
      if (wx + 14 > sx + w - 4) continue;

      // Frame and sill.
      px(wx - 2, winY - 2, 16, 18, P.woodDeep);
      px(wx - 1, winY - 1, 14, 16, P.woodDark);
      px(wx, winY, 12, 14, lit ? P.windowLit : P.windowDark);

      if (lit) {
        // Warm interior with a suggestion of a room behind the glass.
        px(wx, winY, 12, 5, P.windowGlow);
        px(wx + 2, winY + 8, 8, 4, '#e8a94a');
        px(wx - 6, winY + 14, 24, 5, 'rgba(255,207,107,0.16)');
      } else {
        px(wx, winY, 12, 4, '#3d3222');
        px(wx + 1, winY + 1, 4, 3, 'rgba(180,200,220,0.15)');
      }
      // Leaded panes.
      px(wx + 5, winY, 2, 14, P.woodDeep);
      px(wx, winY + 6, 12, 2, P.woodDeep);
      // Shutters.
      px(wx - 6, winY - 1, 4, 16, P.woodDark);
      px(wx + 14, winY - 1, 4, 16, P.woodDark);
      px(wx - 5, winY, 2, 14, P.wood);
      px(wx + 15, winY, 2, 14, P.wood);
      // Sill with a window box of flowers.
      px(wx - 4, winY + 15, 20, 3, P.woodLit);
      px(wx - 2, winY + 17, 16, 5, P.woodDark);
      px(wx - 1, winY + 18, 14, 3, P.trunk);
      for (let f = 0; f < 4; f += 1) {
        const fx = wx + 1 + f * 3;
        px(fx, winY + 16, 2, 2, f % 2 ? '#c8434f' : '#e0a53f');
        px(fx, winY + 15, 1, 1, P.leafHi);
      }
    }

    // ---- chimney -----------------------------------------------------------
    if (!b.tower) {
      const cx = sx + w - 18;
      px(cx - 1, sy - 14, 13, 20, P.stoneDeep);
      for (let row = 0; row < 4; row += 1) {
        px(cx, sy - 13 + row * 4, 11, 3, row % 2 ? P.stone : P.stoneDark);
        px(cx, sy - 13 + row * 4, 11, 1, P.stoneHi);
      }
      px(cx - 2, sy - 17, 15, 4, P.stoneDark);
      px(cx - 2, sy - 17, 15, 1, P.stoneHi);

      if (lit) {
        for (let i = 0; i < 5; i += 1) {
          const t = (clock * 0.7 + i * 0.5) % 2.6;
          const alpha = Math.max(0, 0.34 - t * 0.13);
          const size = 4 + t * 3;
          px(cx + 3 + Math.sin(t * 2.4 + i) * 5, sy - 20 - t * 16, size, size,
            `rgba(196,186,172,${alpha.toFixed(3)})`);
        }
      }
    }

    // ---- tower crenellations ------------------------------------------------
    if (b.tower) {
      for (let i = 0; i * 12 < w + 8; i += 1) {
        if (i % 2) continue;
        const bx = sx - 4 + i * 12;
        px(bx, sy - 12, 11, 12, P.stoneDeep);
        px(bx + 1, sy - 11, 9, 10, P.stone);
        px(bx + 1, sy - 11, 9, 1, P.stoneHi);
      }
      px(sx - 5, sy - 2, w + 10, 4, P.stoneDark);
      px(sx - 5, sy - 2, w + 10, 1, P.stoneHi);
    }

    // ---- forge glow ---------------------------------------------------------
    if (b.forge && lit) {
      const flicker = 0.45 + Math.sin(clock * 11) * 0.18 + Math.sin(clock * 7.3) * 0.1;
      px(dx - 6, dy + 4, doorW + 12, doorH, `rgba(232,118,58,${(flicker * 0.4).toFixed(3)})`);
      px(dx - 2, dy + 12, doorW + 4, doorH - 12, `rgba(255,179,92,${(flicker * 0.5).toFixed(3)})`);
    }

    // ---- theatre marquee ----------------------------------------------------
    if (b.theater) {
      // A lit marquee band over the doors, with chasing bulbs.
      const marqueeY = wallY + 6;
      px(sx - 4, marqueeY, w + 8, 16, P.woodDeep);
      px(sx - 2, marqueeY + 2, w + 4, 12, lit ? '#3a1f38' : '#241a22');
      px(sx - 2, marqueeY + 2, w + 4, 2, P.tilePlumLit);

      const bulbs = Math.floor((w + 8) / 10);
      for (let i = 0; i < bulbs; i += 1) {
        const on = !lit || (Math.floor(clock * 4) + i) % 3 !== 0;
        const bx = sx - 2 + i * 10 + 3;
        px(bx, marqueeY - 3, 4, 4, on ? P.windowGlow : P.woodDark);
        if (on && lit) px(bx - 1, marqueeY - 4, 6, 6, 'rgba(255,230,168,0.22)');
      }

      // Three "letters" on the marquee board, suggested rather than spelled.
      for (let i = 0; i < 4; i += 1) {
        px(sx + 8 + i * 14, marqueeY + 6, 8, 5, lit ? P.gold : P.stoneDark);
      }

      // Playbills pasted either side of the doors.
      [sx + 4, sx + w - 16].forEach((bx, index) => {
        px(bx, wallY + 26, 12, 16, P.parchmentPost || '#d8c49a');
        px(bx, wallY + 26, 12, 1, '#f0e0bc');
        px(bx + 2, wallY + 29, 8, 1, P.woodDeep);
        px(bx + 2, wallY + 32, 6, 1, P.woodDeep);
        px(bx + 2, wallY + 35, 8, 1, P.woodDeep);
        if (index === 0) px(bx + 3, wallY + 38, 5, 2, P.tilePlum);
      });

      // Footlights washing the facade from below.
      if (lit) {
        px(sx - 2, sy + h - 10, w + 4, 8, 'rgba(255,207,107,0.14)');
        for (let i = 0; i < w; i += 12) {
          px(sx + i + 4, sy + h - 5, 5, 3, P.windowGlow);
        }
      }
    }

    // ---- hanging sign -------------------------------------------------------
    const signX = sx + 6;
    px(signX, wallY - 2, 3, 8, P.woodDeep);
    px(signX, wallY + 5, 16, 2, P.woodDeep);
    px(signX + 12, wallY + 6, 2, 4, P.stoneDark);
    px(signX + 6, wallY + 9, 16, 12, P.woodDeep);
    px(signX + 7, wallY + 10, 14, 10, P.wood);
    px(signX + 7, wallY + 10, 14, 1, P.woodHi);
    px(signX + 9, wallY + 13, 10, 1, P.woodDeep);
    px(signX + 9, wallY + 16, 6, 1, P.woodDeep);
  }

  /* ======================================================================
     Props
     ====================================================================== */

  function drawProp(prop) {
    const sx = prop.x * TILE;
    const sy = prop.y * TILE;

    switch (prop.kind) {
      case 'tree':
        drawTree(sx, sy, prop.seed);
        break;
      case 'bush':
        drawBush(sx, sy, prop.seed);
        break;
      case 'barrel':
        drawBarrel(sx, sy);
        break;
      case 'well':
        drawWell(sx, sy);
        break;
      case 'picnic':
        drawPicnic(sx, sy);
        break;
      case 'crates':
        drawCrates(sx, sy);
        break;
      case 'stump':
        drawStump(sx, sy);
        break;
      case 'lamp':
        drawLamp(sx, sy);
        break;
      case 'sign':
        drawSignpost(sx, sy);
        break;
      case 'fence':
        drawFence(sx, sy);
        break;
      case 'flowers':
        drawFlowers(sx, sy, prop.seed);
        break;
      case 'cart':
        drawCart(sx, sy);
        break;
      default:
        break;
    }
  }

  /**
   * A broadleaf tree: layered canopy blobs so the silhouette is irregular, a
   * tapering trunk with bark strokes, and a sway that only moves the crown.
   */
  function drawTree(sx, sy, seed) {
    const sway = Math.round(Math.sin(clock * 0.7 + seed * 6) * 1.6);
    const cx = sx + 12;

    px(sx + 3, sy + 17, 18, 6, P.shadow);

    // Trunk with a root flare.
    px(cx - 3, sy + 4, 6, 19, P.trunkDark);
    px(cx - 2, sy + 4, 3, 18, P.trunk);
    px(cx - 1, sy + 6, 1, 14, P.trunkLit);
    px(cx - 5, sy + 20, 4, 3, P.trunkDark);
    px(cx + 1, sy + 20, 4, 3, P.trunkDark);
    px(cx - 4, sy + 12, 2, 2, P.trunkDark);
    px(cx + 2, sy + 9, 2, 2, P.trunkDark);

    // Canopy: five overlapping clusters, dark to light.
    const blobs = [
      [-12, -8, 16, 14, P.leafDeep],
      [2, -10, 15, 15, P.leafDeep],
      [-9, -14, 14, 13, P.leafDark],
      [1, -16, 13, 12, P.leafDark],
      [-5, -12, 14, 12, P.leaf],
      [-2, -18, 10, 9, P.leaf],
    ];
    blobs.forEach(([dx, dy, w, h, colour]) => {
      px(cx + dx + sway, sy + dy + 8, w, h, colour);
    });

    // Lit crown and a few leaf specks catching the sun.
    px(cx - 6 + sway, sy - 8, 9, 6, P.leafLit);
    px(cx - 1 + sway, sy - 10, 6, 4, P.leafHi);
    px(cx + 4 + sway, sy - 2, 5, 4, P.leafLit);
    px(cx - 10 + sway, sy + 2, 4, 3, P.leafLit);
    // Underside shadow where the canopy meets the trunk.
    px(cx - 8 + sway, sy + 8, 16, 3, P.leafDeep);
  }

  function drawBush(sx, sy, seed) {
    const sway = Math.round(Math.sin(clock * 0.9 + seed * 4) * 1);
    const random = seeded(Math.floor(seed * 900) + 11);
    px(sx + 4, sy + 17, 16, 5, P.shadow);

    // Three overlapping lobes give an irregular silhouette.
    px(sx + 2 + sway, sy + 9, 12, 11, P.leafDeep);
    px(sx + 10 + sway, sy + 8, 12, 12, P.leafDeep);
    px(sx + 5 + sway, sy + 5, 14, 12, P.leafDark);
    px(sx + 9 + sway, sy + 4, 9, 9, P.leaf);
    px(sx + 3 + sway, sy + 11, 8, 7, P.leafDark);

    // Lit crown and scattered leaf specks.
    px(sx + 8 + sway, sy + 4, 6, 4, P.leafLit);
    px(sx + 10 + sway, sy + 3, 3, 2, P.leafHi);
    px(sx + 15 + sway, sy + 9, 4, 3, P.leafLit);
    px(sx + 4 + sway, sy + 8, 3, 3, P.leafLit);
    for (let i = 0; i < 5; i += 1) {
      px(sx + 3 + Math.floor(random() * 17) + sway, sy + 5 + Math.floor(random() * 13),
        1, 1, random() > 0.5 ? P.leafHi : P.leafDeep);
    }

    // A few twigs poking out of the base.
    px(sx + 8, sy + 18, 1, 3, P.trunkDark);
    px(sx + 13, sy + 18, 1, 3, P.trunkDark);

    // Berries on about half the bushes.
    if (seed > 0.5) {
      [[6, 12], [15, 14], [10, 9], [12, 15]].forEach(([bx, by]) => {
        px(sx + bx + sway, sy + by, 2, 2, '#c8434f');
        px(sx + bx + sway, sy + by, 1, 1, '#e05a66');
      });
    }
  }

  /** A banded barrel, staved and hooped. */
  function drawBarrel(sx, sy) {
    px(sx + 5, sy + 19, 15, 4, P.shadow);
    px(sx + 5, sy + 4, 14, 18, P.woodDeep);
    // Staves bulge in the middle: wider band through the centre.
    for (let i = 0; i < 14; i += 3) {
      px(sx + 6 + i, sy + 5, 2, 16, i % 6 ? P.wood : P.woodDark);
    }
    px(sx + 4, sy + 8, 16, 3, P.stoneDark);
    px(sx + 4, sy + 8, 16, 1, P.stoneHi);
    px(sx + 4, sy + 16, 16, 3, P.stoneDark);
    px(sx + 4, sy + 16, 16, 1, P.stoneHi);
    // Lid.
    px(sx + 5, sy + 3, 14, 3, P.woodLit);
    px(sx + 7, sy + 3, 10, 1, P.woodHi);
  }

  function drawWell(sx, sy) {
    px(sx + 1, sy + 16, 22, 6, P.shadow);
    // Stone drum, coursed.
    px(sx + 2, sy + 6, 20, 16, P.stoneDeep);
    for (let row = 0; row < 3; row += 1) {
      const offset = row % 2 ? 4 : 0;
      for (let i = -offset; i < 20; i += 8) {
        px(sx + 3 + Math.max(0, i), sy + 7 + row * 5, Math.min(7, 19 - i), 4,
          row % 2 ? P.stone : P.stoneDark);
        px(sx + 3 + Math.max(0, i), sy + 7 + row * 5, Math.min(7, 19 - i), 1, P.stoneHi);
      }
    }
    // Dark water with a glint.
    px(sx + 6, sy + 8, 12, 6, '#16323a');
    px(sx + 8, sy + 9, 4, 2, P.waterLit);
    // Posts and a shingled canopy.
    px(sx + 3, sy - 10, 4, 17, P.woodDark);
    px(sx + 17, sy - 10, 4, 17, P.woodDark);
    px(sx + 4, sy - 9, 1, 15, P.woodLit);
    for (let c = 0; c < 3; c += 1) {
      const inset = c * 2;
      px(sx + inset, sy - 14 + c * 3, 24 - inset * 2, 4, c === 0 ? P.slateLit : P.slate);
      px(sx + inset, sy - 14 + c * 3, 24 - inset * 2, 1, P.slateLit);
    }
    // Winch, rope and bucket.
    px(sx + 5, sy - 4, 14, 3, P.woodLit);
    px(sx + 11, sy - 1, 1, 7, '#8a7250');
    px(sx + 9, sy + 5, 6, 5, P.woodDark);
    px(sx + 9, sy + 5, 6, 1, P.stoneHi);
  }

  function drawPicnic(sx, sy) {
    px(sx - 1, sy + 15, 26, 6, P.shadow);
    // Benches either side, drawn first so the table overlaps them.
    px(sx - 2, sy + 12, 28, 3, P.woodDark);
    px(sx - 2, sy + 12, 28, 1, P.woodLit);
    // Table top, planked.
    px(sx - 3, sy + 5, 30, 5, P.woodDeep);
    for (let i = 0; i < 30; i += 6) {
      px(sx - 2 + i, sy + 6, 5, 3, i % 12 ? P.wood : P.woodLit);
    }
    px(sx - 3, sy + 5, 30, 1, P.woodHi);
    // Legs, splayed.
    px(sx + 1, sy + 10, 3, 11, P.woodDark);
    px(sx + 20, sy + 10, 3, 11, P.woodDark);
    // Things left on it: a cloth, a loaf, a mug, an apple.
    px(sx + 3, sy + 2, 9, 4, '#c8434f');
    px(sx + 3, sy + 2, 9, 1, '#e05a66');
    px(sx + 14, sy + 1, 6, 5, '#c8a45c');
    px(sx + 14, sy + 1, 6, 1, '#e0c47a');
    px(sx + 21, sy + 2, 4, 4, P.plasterLit);
    px(sx + 25, sy + 3, 1, 2, P.plasterLit);
  }

  function drawCrates(sx, sy) {
    px(sx + 1, sy + 17, 22, 5, P.shadow);
    const crate = (x, y, size) => {
      px(x, y, size, size, P.woodDeep);
      px(x + 1, y + 1, size - 2, size - 2, P.wood);
      px(x + 1, y + 1, size - 2, 1, P.woodHi);
      // Cross-brace.
      for (let i = 1; i < size - 1; i += 1) {
        px(x + i, y + i, 1, 1, P.woodDark);
        px(x + size - 1 - i, y + i, 1, 1, P.woodDark);
      }
    };
    crate(sx + 1, sy + 6, 12);
    crate(sx + 12, sy + 10, 10);
    crate(sx + 4, sy - 2, 9);
    // A sack leaning against them.
    px(sx + 15, sy + 2, 8, 9, '#a89058');
    px(sx + 16, sy + 1, 6, 3, '#c0a86c');
    px(sx + 17, sy + 5, 3, 3, '#8a7248');
  }

  function drawStump(sx, sy) {
    px(sx + 4, sy + 16, 16, 5, P.shadow);
    px(sx + 5, sy + 8, 14, 11, P.trunkDark);
    px(sx + 6, sy + 9, 12, 9, P.trunk);
    // Rings on the cut face.
    px(sx + 5, sy + 6, 14, 4, P.trunkLit);
    px(sx + 8, sy + 7, 8, 2, P.trunk);
    px(sx + 10, sy + 7, 4, 2, P.trunkDark);
    px(sx + 11, sy + 8, 2, 1, P.trunkLit);
    // The axe buried in it.
    px(sx + 13, sy - 4, 2, 11, P.wood);
    px(sx + 13, sy - 4, 1, 11, P.woodHi);
    px(sx + 10, sy - 6, 7, 4, P.stone);
    px(sx + 10, sy - 6, 7, 1, P.stoneHi);
    px(sx + 9, sy - 5, 2, 3, P.stoneDark);
    // Split logs beside it.
    px(sx - 2, sy + 15, 9, 5, P.trunkDark);
    px(sx - 1, sy + 16, 3, 3, P.trunkLit);
  }

  function drawLamp(sx, sy) {
    const glow = 0.6 + Math.sin(clock * 2.4 + sx) * 0.16;
    px(sx + 8, sy + 20, 10, 4, P.shadow);
    // Post with a base.
    px(sx + 9, sy + 16, 8, 5, P.stoneDeep);
    px(sx + 10, sy + 17, 6, 3, P.stoneDark);
    px(sx + 11, sy - 4, 4, 21, P.outline);
    px(sx + 12, sy - 3, 2, 19, P.stoneDark);
    // Lantern housing.
    px(sx + 7, sy - 12, 12, 10, P.outline);
    px(sx + 8, sy - 11, 10, 8, `rgba(255,207,107,${glow.toFixed(2)})`);
    px(sx + 9, sy - 10, 8, 3, P.windowGlow);
    px(sx + 12, sy - 8, 2, 4, '#fff4d0');
    // Cap and finial.
    px(sx + 6, sy - 15, 14, 4, P.stoneDeep);
    px(sx + 6, sy - 15, 14, 1, P.stoneHi);
    px(sx + 12, sy - 18, 2, 3, P.stoneDark);
    // Pooled light on the ground.
    px(sx - 2, sy + 6, 28, 16, `rgba(255,190,90,${(glow * 0.07).toFixed(3)})`);
    px(sx + 2, sy + 12, 20, 8, `rgba(255,190,90,${(glow * 0.06).toFixed(3)})`);
  }

  function drawSignpost(sx, sy) {
    px(sx + 6, sy + 18, 14, 4, P.shadow);
    px(sx + 11, sy + 2, 4, 18, P.woodDeep);
    px(sx + 12, sy + 3, 2, 16, P.wood);
    // Two arms pointing opposite ways.
    px(sx - 2, sy + 1, 18, 8, P.woodDeep);
    px(sx - 1, sy + 2, 16, 6, P.wood);
    px(sx - 1, sy + 2, 16, 1, P.woodHi);
    px(sx + 2, sy + 4, 9, 1, P.woodDeep);
    px(sx + 2, sy + 6, 6, 1, P.woodDeep);
    px(sx + 10, sy + 11, 16, 7, P.woodDeep);
    px(sx + 11, sy + 12, 14, 5, P.woodLit);
    px(sx + 13, sy + 14, 8, 1, P.woodDeep);
  }

  function drawFence(sx, sy) {
    px(sx, sy + 16, TILE, 4, P.shadowSoft);
    // Two rails and a post.
    px(sx - 2, sy + 6, TILE + 4, 3, P.woodDark);
    px(sx - 2, sy + 6, TILE + 4, 1, P.woodLit);
    px(sx - 2, sy + 12, TILE + 4, 3, P.woodDark);
    px(sx - 2, sy + 12, TILE + 4, 1, P.woodLit);
    px(sx + 9, sy + 2, 5, 17, P.woodDeep);
    px(sx + 10, sy + 3, 3, 15, P.wood);
    px(sx + 10, sy + 3, 3, 1, P.woodHi);
  }

  function drawFlowers(sx, sy, seed) {
    const random = seeded(Math.floor(seed * 1000) + 3);
    for (let i = 0; i < 6; i += 1) {
      const x = sx + 2 + Math.floor(random() * 19);
      const y = sy + 6 + Math.floor(random() * 14);
      const sway = Math.round(Math.sin(clock * 1.3 + i) * 0.6);
      px(x, y + 2, 1, 4, P.grassDeep);
      const petal = ['#d8555f', '#e0a53f', '#c88ad0', '#f0e08a'][i % 4];
      px(x - 1 + sway, y, 3, 2, petal);
      px(x + sway, y - 1, 1, 1, petal);
      px(x + sway, y, 1, 1, P.windowGlow);
    }
  }

  /** A hand cart, parked outside the market. */
  function drawCart(sx, sy) {
    px(sx, sy + 17, 26, 5, P.shadow);
    // Bed and sideboards.
    px(sx + 1, sy + 6, 24, 9, P.woodDeep);
    for (let i = 0; i < 24; i += 5) {
      px(sx + 2 + i, sy + 7, 4, 7, i % 10 ? P.wood : P.woodDark);
    }
    px(sx + 1, sy + 6, 24, 1, P.woodHi);
    // Wheels with spokes.
    [sx + 5, sx + 18].forEach((wx) => {
      px(wx - 4, sy + 12, 9, 9, P.woodDeep);
      px(wx - 3, sy + 13, 7, 7, P.trunk);
      px(wx - 1, sy + 13, 2, 7, P.woodLit);
      px(wx - 3, sy + 16, 7, 2, P.woodLit);
      px(wx - 1, sy + 16, 2, 2, P.stoneDark);
    });
    // Handle and a load of produce.
    px(sx + 24, sy + 4, 8, 2, P.woodDark);
    px(sx + 6, sy + 2, 5, 5, '#c8434f');
    px(sx + 12, sy + 3, 4, 4, '#e0a53f');
    px(sx + 17, sy + 2, 5, 5, '#6d9a3c');
  }

  /* ======================================================================
     Camera
     ====================================================================== */

  const camera = {
    x: WORLD_W / 2,
    y: WORLD_H / 2,
    zoom: 1.5,
    targetX: WORLD_W / 2,
    targetY: WORLD_H / 2,
    follow: true,
    followId: null,

    /** Keep the view inside the world, allowing for the zoom level. */
    clamp() {
      const halfW = canvas.width / (2 * this.zoom);
      const halfH = canvas.height / (2 * this.zoom);
      this.targetX = Math.max(halfW, Math.min(WORLD_W - halfW, this.targetX));
      this.targetY = Math.max(halfH, Math.min(WORLD_H - halfH, this.targetY));
      if (WORLD_W * this.zoom < canvas.width) this.targetX = WORLD_W / 2;
      if (WORLD_H * this.zoom < canvas.height) this.targetY = WORLD_H / 2;
    },

    update(dt) {
      this.clamp();
      // Critically-damped-ish follow; snappy without overshoot.
      const k = 1 - Math.pow(0.001, dt);
      this.x += (this.targetX - this.x) * k;
      this.y += (this.targetY - this.y) * k;
    },

    centreOn(x, y) {
      this.targetX = x;
      this.targetY = y;
      this.clamp();
    },

    /** Screen -> world. */
    toWorld(sx, sy) {
      return {
        x: (sx - canvas.width / 2) / this.zoom + this.x,
        y: (sy - canvas.height / 2) / this.zoom + this.y,
      };
    },

    /** World -> screen. */
    toScreen(wx, wy) {
      return {
        x: (wx - this.x) * this.zoom + canvas.width / 2,
        y: (wy - this.y) * this.zoom + canvas.height / 2,
      };
    },
  };

  /* ======================================================================
     State
     ====================================================================== */

  let villagers = [];
  const villagerById = new Map();
  const activeBuildings = new Set();
  let clock = 0;
  let lastFrame = performance.now();
  let socket = null;
  let reconnectDelay = 1000;
  let selectedBuilding = null;
  /** The Bard's most recent videos, newest first. */
  let shorts = [];

  /* ======================================================================
     Render loop
     ====================================================================== */

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(320, Math.round(rect.width * dpr));
    canvas.height = Math.max(240, Math.round(rect.height * dpr));
    ctx.imageSmoothingEnabled = false;
    // Fit the world in view by default, then let the user zoom.
    const fit = Math.min(canvas.width / WORLD_W, canvas.height / WORLD_H);
    camera.zoom = Math.max(fit, Math.min(camera.zoom, 4));
    camera.clamp();
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;
    clock += dt;

    villagers.forEach((v) => v.update(dt));

    // Follow whoever is working, if following is on.
    if (camera.follow) {
      const working = villagers.find((v) => v.status === 'working');
      const target = working || villagerById.get(camera.followId);
      if (target) camera.centreOn(target.x, target.y);
    }
    camera.update(dt);

    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#1d3b2a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-camera.x, -camera.y);

    drawGround();

    // Depth sort every world object by its base y, so a villager standing
    // below a building overlaps it and one behind is hidden.
    const drawables = [];
    BUILDINGS.forEach((b) =>
      drawables.push({ y: (b.y + b.h) * TILE, draw: () => drawBuilding(b) }),
    );
    PROPS.forEach((prop) =>
      drawables.push({ y: prop.y * TILE + TILE, draw: () => drawProp(prop) }),
    );
    villagers.forEach((v) =>
      drawables.push({
        y: v.y + 2,
        draw: () => {
          drawVillagerSprite(ctx, v, {
            x: v.x,
            y: v.y,
            scale: 1,
            frame: v.frame,
            facing: v.facing,
            pose: v.pose,
          });
          drawWorkEffects(v);
        },
      }),
    );

    drawables.sort((a, b) => a.y - b.y);
    drawables.forEach((item) => item.draw());

    ctx.restore();

    positionNametags();
    requestAnimationFrame(frame);
  }

  /** Sparks, ink, sound rings — the flourish that sells the work state. */
  function drawWorkEffects(v) {
    if (v.status !== 'working' && v.mode !== 'working') {
      if (v.emoteFor > 0 && v.emote) drawEmote(v);
      return;
    }

    // Effects hang off the held prop, which sits at roughly two thirds of the
    // sprite's height.
    const handX = v.x + (v.facing > 0 ? 11 : -11);
    const handY = v.y - 18;
    const pose = v.pose;

    if (pose === 'hammer') {
      // Sparks arc off the anvil and fall, with a hot core and a cooling tail.
      for (let i = 0; i < 8; i += 1) {
        const t = (clock * 3.4 + i * 0.31) % 1;
        const angle = -0.4 - i * 0.28;
        const speed = 16 + (i % 3) * 7;
        const sx = handX + Math.cos(angle) * t * speed * v.facing;
        const sy = handY - Math.sin(angle) * t * speed + t * t * 22;
        const colour = t < 0.3 ? '#fff4d0' : t < 0.6 ? P.emberHi : P.ember;
        px(sx, sy, t < 0.5 ? 2 : 1, t < 0.5 ? 2 : 1, colour);
      }
      // The anvil's glow pulses on the strike frames.
      if (v.frame % 2 === 0) {
        px(handX - 6, handY + 4, 12, 3, 'rgba(255,179,92,0.35)');
      }
    } else if (pose === 'quill') {
      // Ink motes lifting off the page.
      for (let i = 0; i < 3; i += 1) {
        const t = (clock * 1.4 + i * 0.4) % 1;
        px(handX + Math.sin(t * 6 + i) * 3, handY - 4 - t * 12, 2, 2,
          `rgba(90,74,125,${(1 - t).toFixed(2)})`);
      }
      // A written line appearing beneath the quill.
      px(handX - 8, handY + 9, Math.round(((clock * 8) % 10)) + 2, 1, '#5a4a7d');
    } else if (pose === 'bell') {
      // Concentric rings, two out of phase.
      for (let i = 0; i < 2; i += 1) {
        const t = ((clock * 1.6 + i * 0.5) % 1);
        ctx.strokeStyle = `rgba(224,165,63,${((1 - t) * 0.75).toFixed(2)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(handX, handY + 6, 5 + t * 26, 0, Math.PI * 2);
        ctx.stroke();
      }
    } else if (pose === 'spyglass') {
      // A glint travelling out along the sightline.
      const t = (clock * 1.1) % 1;
      px(handX + (10 + t * 22) * v.facing, handY + 5, 2, 1,
        `rgba(168,212,255,${(1 - t).toFixed(2)})`);
      px(handX + 8 * v.facing, handY + 4, 2, 2, 'rgba(255,255,255,0.7)');
    } else if (pose === 'inspect') {
      // The lens brightens and dims as it sweeps.
      const pulse = 0.28 + Math.sin(clock * 4) * 0.16;
      px(handX - 2, handY, 10, 10, `rgba(168,212,255,${pulse.toFixed(2)})`);
      px(handX + 1, handY + 3, 3, 3, `rgba(255,255,255,${(pulse + 0.2).toFixed(2)})`);
    } else if (pose === 'trade') {
      // Coins rising from the crate.
      for (let i = 0; i < 3; i += 1) {
        const t = (clock * 1.2 + i * 0.33) % 1;
        px(handX + Math.sin(t * 5 + i) * 4, handY - t * 18, 3, 3,
          `rgba(245,205,106,${(1 - t).toFixed(2)})`);
        px(handX + Math.sin(t * 5 + i) * 4, handY - t * 18, 1, 1,
          `rgba(255,244,208,${(1 - t).toFixed(2)})`);
      }
    } else if (pose === 'lute') {
      // Musical notes drifting up and out as he composes the narration.
      for (let i = 0; i < 4; i += 1) {
        const t = (clock * 0.9 + i * 0.28) % 1;
        const nx = handX + Math.sin(t * 4 + i * 2) * 9;
        const ny = handY - 4 - t * 26;
        const alpha = (1 - t).toFixed(2);
        px(nx, ny, 3, 3, `rgba(240,224,188,${alpha})`);
        px(nx + 2, ny - 4, 1, 5, `rgba(240,224,188,${alpha})`);
        px(nx + 2, ny - 5, 3, 2, `rgba(224,165,63,${alpha})`);
      }
    } else if (pose === 'ponder') {
      // Thought puffs drifting up and growing.
      for (let i = 0; i < 3; i += 1) {
        const t = (clock * 0.7 + i * 0.34) % 1;
        const size = 2 + t * 4;
        px(v.x + 8 + t * 5, v.y - 36 - t * 14, size, size,
          `rgba(242,230,200,${((1 - t) * 0.55).toFixed(2)})`);
      }
    } else if (pose === 'chop') {
      if (v.frame % 2 === 1) {
        for (let i = 0; i < 4; i += 1) {
          const t = (clock * 4 + i * 0.25) % 1;
          px(handX + i * 3 * v.facing, handY + 8 + t * 10, 2, 2,
            `rgba(117,80,58,${(1 - t).toFixed(2)})`);
        }
      }
    }

    if (v.emoteFor > 0 && v.emote) drawEmote(v);
  }

  function drawEmote(v) {
    const lift = Math.min(1, (2.2 - v.emoteFor) * 4);
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, v.emoteFor));
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(v.emote, v.x, v.y - 42 - lift * 5);
    ctx.restore();
  }

  /* ======================================================================
     Name tags (DOM overlay, so the text stays sharp)
     ====================================================================== */

  const tagNodes = new Map();

  function positionNametags() {
    villagers.forEach((v) => {
      let node = tagNodes.get(v.id);
      if (!node) {
        node = document.createElement('div');
        node.className = 'nametag';
        nametagLayer.appendChild(node);
        tagNodes.set(v.id, node);
      }

      const screen = camera.toScreen(v.x, v.y - 38);
      const visible =
        screen.x > -80 && screen.x < canvas.clientWidth + 80 && screen.y > -40 && screen.y < canvas.clientHeight + 40;
      node.style.display = visible ? 'block' : 'none';
      if (!visible) return;

      // Canvas is drawn at devicePixelRatio; the overlay works in CSS pixels.
      const ratio = canvas.clientWidth / canvas.width;
      node.style.left = screen.x * ratio + 'px';
      node.style.top = screen.y * ratio + 'px';

      const working = v.status === 'working';
      node.className = 'nametag' + (working ? ' nametag--working' : '');
      node.textContent = (working ? '⚙ ' : '') + v.name.split(' ')[0];
    });
  }

  /* ======================================================================
     Interaction
     ====================================================================== */

  let pointer = { down: false, moved: false, x: 0, y: 0 };

  function eventToCanvas(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  canvas.addEventListener('pointerdown', (event) => {
    pointer = { down: true, moved: false, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('is-panning');
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!pointer.down) return;
    const dx = event.clientX - pointer.x;
    const dy = event.clientY - pointer.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) pointer.moved = true;

    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    camera.targetX -= (dx * scale) / camera.zoom;
    camera.targetY -= (dy * scale) / camera.zoom;
    camera.x = camera.targetX;
    camera.y = camera.targetY;
    camera.follow = false;
    updateFollowLabel();
    camera.clamp();
    pointer.x = event.clientX;
    pointer.y = event.clientY;
  });

  canvas.addEventListener('pointerup', (event) => {
    canvas.classList.remove('is-panning');
    pointer.down = false;
    if (pointer.moved) return;
    handleClick(eventToCanvas(event));
  });

  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const fit = Math.min(canvas.width / WORLD_W, canvas.height / WORLD_H);
    camera.zoom = Math.max(fit, Math.min(4.5, camera.zoom * factor));
    camera.clamp();
  }, { passive: false });

  /** Villager first, then building — a villager standing in a doorway wins. */
  function handleClick(point) {
    const world = camera.toWorld(point.x, point.y);

    // The sprite is a tall rectangle standing on its feet, so hit-test an
    // upright box rather than a circle — a circle either misses the head or
    // swallows the neighbouring tile.
    let hit = null;
    let bestDistance = Infinity;
    villagers.forEach((v) => {
      const dx = Math.abs(v.x - world.x);
      const dy = world.y - (v.y - 32);
      if (dx > 10 || dy < 0 || dy > 34) return;
      const d = dx + Math.abs(dy - 17);
      if (d < bestDistance) {
        bestDistance = d;
        hit = v;
      }
    });

    if (hit) {
      openVillager(hit.id);
      return;
    }

    const tx = Math.floor(world.x / TILE);
    const ty = Math.floor(world.y / TILE);
    const building = BUILDINGS.find(
      (b) => tx >= b.x && tx < b.x + b.w && ty >= b.y - 1 && ty < b.y + b.h,
    );
    if (building) openBuilding(building);
  }

  /* ======================================================================
     Dialogue wiring
     ====================================================================== */

  function portraitDrawer(villager) {
    return (target, size) => {
      target.imageSmoothingEnabled = false;
      // A soft vignette behind the bust.
      target.fillStyle = shade(villager.color || '#333', -0.65);
      target.fillRect(0, 0, size, size);
      target.fillStyle = 'rgba(255,255,255,0.05)';
      target.fillRect(0, size * 0.55, size, size * 0.45);
      drawVillagerSprite(target, villager, {
        x: size / 2,
        y: size - 4,
        scale: 1.9,
        frame: 0,
        facing: 1,
        pose: 'idle',
      });
    };
  }

  /**
   * Build the Bard's video preview card. Shown inside the dialogue instead of
   * the generic "last output" block, because a Short is worth watching, not
   * reading a summary of.
   */
  function bardPreview() {
    const latest = shorts[0];
    if (!latest) {
      return (
        '<p class="preview__line preview__muted">No Shorts yet. ' +
        'Press <strong>Generate New Short</strong> and the Bard will write one.</p>'
      );
    }

    const esc = window.escapeHtml;
    const storyboard = latest.render_backend === 'storyboard';
    const silent = latest.voice_backend === 'placeholder';
    const media = storyboard
      ? `<img class="short__media" src="/api/shorts/${latest.id}/video" alt="Storyboard" />`
      : `<video class="short__media" src="/api/shorts/${latest.id}/video" controls playsinline
           preload="metadata" ${latest.thumbnail_path ? `poster="/api/shorts/${latest.id}/thumbnail"` : ''}></video>`;

    const warnings = [];
    if (storyboard) warnings.push('storyboard only — no video toolchain');
    if (silent) warnings.push('silent placeholder track — no TTS available');

    return (
      `<div class="short">` +
      media +
      `<div class="short__meta">` +
      `<p class="short__title">${esc(latest.title)}</p>` +
      `<p class="short__sub">${esc(latest.category)} · ${Math.round(latest.duration_seconds)}s · ` +
      `${esc(latest.status)}</p>` +
      `<p class="short__hook">“${esc(latest.hook)}”</p>` +
      (warnings.length
        ? `<p class="short__warn">⚠️ ${esc(warnings.join(' · '))}</p>`
        : '') +
      `<p class="short__tags">${latest.hashtags.slice(0, 6).map((t) => '#' + esc(t)).join(' ')}</p>` +
      `</div></div>`
    );
  }

  /** Approve / reject buttons for the Bard's newest Short. */
  function bardActions(actions) {
    const latest = shorts[0];
    if (!latest || latest.status !== 'PENDING_APPROVAL') return actions;
    return [
      { id: 'short_approve', label: '✅ Approve & Upload', description: 'Mark it ready to upload' },
      { id: 'short_reject', label: '🚫 Reject', description: 'Discard this Short' },
      ...actions,
    ];
  }

  function openVillager(agentId) {
    const v = villagerById.get(agentId);
    if (!v) return;
    window.Hud.markActive(agentId);

    const line = v.status === 'working' && v.task
      ? v.task
      : v.status === 'error'
        ? `${v.task} ${v.detail}`.trim()
        : v.task || v.idleLine;

    window.Dialogue.show({
      subjectId: agentId,
      name: `${v.emoji} ${v.name}`,
      title: v.title,
      status: v.status,
      text: line,
      output: agentId === 'bard' ? null : v.lastOutput,
      html: agentId === 'bard' ? bardPreview() : null,
      actions: agentId === 'bard' ? bardActions(v.actions || []) : v.actions || [],
      drawPortrait: portraitDrawer(v),
      onAction: (actionId) => triggerAction(agentId, actionId),
    });
  }

  function openBuilding(building) {
    selectedBuilding = building.id;
    const resident = villagers.find((v) => v.home === building.id);
    const text = resident
      ? `${building.name}. ${resident.name} works here — ${resident.status === 'working' ? resident.task : 'currently out and about.'}`
      : `${building.name}. Quiet today.`;

    window.Dialogue.show({
      subjectId: 'building:' + building.id,
      name: `${building.icon} ${building.name}`,
      title: resident ? `Home of ${resident.name}` : 'Village building',
      status: resident ? resident.status : 'idle',
      text,
      output: resident && resident.id !== 'bard' ? resident.lastOutput : null,
      html: building.id === 'theater' ? bardPreview() : null,
      actions: resident
        ? resident.id === 'bard'
          ? bardActions(resident.actions || [])
          : resident.actions || []
        : [],
      drawPortrait: (target, size) => {
        target.fillStyle = '#0d1119';
        target.fillRect(0, 0, size, size);
        target.font = `${Math.round(size * 0.55)}px sans-serif`;
        target.textAlign = 'center';
        target.textBaseline = 'middle';
        target.fillText(building.icon, size / 2, size / 2 + 2);
      },
      onAction: (actionId) => resident && triggerAction(resident.id, actionId),
    });
  }

  async function triggerAction(agentId, actionId) {
    // The Bard's approve/reject act on a Short, not on the villager.
    if (actionId === 'short_approve' || actionId === 'short_reject') {
      const latest = shorts[0];
      if (!latest) return;
      const verb = actionId === 'short_approve' ? 'approve' : 'reject';
      try {
        const response = await fetch(`/api/shorts/${latest.id}/${verb}`, { method: 'POST' });
        const data = await response.json();
        if (data.ok === false && data.accepted === undefined) {
          window.Hud.toast(`Could not ${verb} short #${latest.id}.`, 'warn');
        } else {
          window.Hud.toast(`Short #${latest.id}: ${verb} sent.`, 'success');
          window.Dialogue.hide();
        }
      } catch (error) {
        window.Hud.toast('Request failed: ' + error.message, 'error');
      }
      return;
    }

    try {
      const response = await fetch(`/api/agents/${agentId}/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: actionId }),
      });
      const data = await response.json();

      if (actionId === 'review_logs') {
        const lines = (data.logs || [])
          .slice(-8)
          .reverse()
          .map((entry) => `• ${entry.status}: ${entry.task || '—'}`)
          .join('\n');
        window.Dialogue.type(lines || 'No activity recorded yet.');
        return;
      }

      if (!response.ok || data.accepted === false) {
        window.Sfx.error();
        window.Hud.toast(data.reason || data.detail || 'The village refused that.', 'warn');
        return;
      }
      window.Hud.toast(`${agentId}: ${actionId} started.`, 'info');
      window.Dialogue.hide();
    } catch (error) {
      window.Sfx.error();
      window.Hud.toast('Could not reach the village: ' + error.message, 'error');
    }
  }

  /* ======================================================================
     WebSocket
     ====================================================================== */

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${location.host}/ws`);
    window.Hud.setConnection('connecting');

    socket.onopen = () => {
      reconnectDelay = 1000;
      window.Hud.setConnection('connected');
      window.Hud.log('Connected to Oakhaven.', 'success');
    };

    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      handleMessage(message);
    };

    socket.onclose = () => {
      window.Hud.setConnection('disconnected');
      window.Hud.log(`Connection lost — retrying in ${Math.round(reconnectDelay / 1000)}s.`, 'warn');
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 30000);
    };

    socket.onerror = () => socket.close();
  }

  function handleMessage(message) {
    switch (message.type) {
      case 'snapshot':
        applySnapshot(message.data);
        break;

      case 'agent_state': {
        const agent = message.agent;
        const v = villagerById.get(agent.id);
        if (v) v.setState(agent);
        window.Hud.updateAgent(agent);
        refreshActiveBuildings();
        if (agent.status === 'working') window.Hud.log(`${agent.name}: ${agent.task}`, 'info');
        if (agent.status === 'error') window.Hud.log(`${agent.name}: ${agent.detail}`, 'error');
        if (window.Dialogue.subject === agent.id) {
          window.Dialogue.update({
            subjectId: agent.id,
            status: agent.status,
            text: agent.task,
            output: v ? v.lastOutput : null,
          });
        }
        break;
      }

      case 'agent_output': {
        const v = villagerById.get(message.agent_id);
        if (v) v.lastOutput = message.output;
        if (window.Dialogue.subject === message.agent_id) {
          window.Dialogue.update({ subjectId: message.agent_id, output: message.output });
        }
        break;
      }

      case 'listing':
        if (message.listing && message.listing.id) {
          window.Hud.upsertListing(message.listing);
        }
        break;

      case 'edition':
        // The Ledger's lifecycle, in the Chronicle. No popup: the approval
        // card in Telegram is where a decision is actually asked for.
        if (message.edition && message.edition.id) {
          const ed = message.edition;
          const state = String(ed.status || '').toLowerCase();
          window.Hud.log(
            `Morning Ledger #${ed.id} ${state}: ${ed.title || 'untitled'}`,
            state === 'published' ? 'success' : state === 'rejected' ? 'warn' : 'info',
          );
        }
        break;

      case 'deal':
        // Curated, approved, rejected, pinned or re-measured — every one of
        // those mutates the row, so the tab re-reads rather than guessing how
        // to merge a partial update into what it already shows.
        if (message.deal && message.deal.id) {
          window.Hud.log(
            `Scout: ${message.deal.product || 'a deal'} (${String(message.deal.status || '').toLowerCase()})`,
            'info',
          );
          refreshDeals();
        }
        break;

      case 'stats':
        window.Hud.updateStats(message.stats);
        break;

      case 'shorts':
        shorts = message.shorts || [];
        window.Hud.renderShorts(shorts);
        if (window.Dialogue.subject === 'bard' || window.Dialogue.subject === 'building:theater') {
          openVillager('bard');
        }
        break;

      case 'pipeline':
        if (message.stage === 'run_started') window.Hud.log('A new village run begins.', 'success');
        if (message.stage === 'run_finished') {
          window.Hud.log(
            message.ok ? `Run complete — listing #${message.listing_id}.` : 'Run ended with findings.',
            message.ok ? 'success' : 'warn',
          );
        }
        break;

      case 'toast':
        window.Hud.toast(message.message, message.tone);
        window.Hud.log(message.message, message.tone);
        break;

      case 'log':
        // Chronicle only, no popup: routine progress that is worth a record
        // but not worth interrupting anyone for.
        window.Hud.log(message.message, message.tone || 'info');
        break;

      default:
        break;
    }
  }

  function applySnapshot(data) {
    // Villagers are rebuilt only on the first snapshot; later ones just sync
    // state, so a reconnect does not teleport everyone home.
    if (!villagers.length) {
      villagers = data.villagers.map((info) => new Villager(info));
      villagers.forEach((v) => villagerById.set(v.id, v));
      window.Hud.renderRoster(data.villagers, (target, id, size) => {
        const villager = villagerById.get(id);
        if (!villager) return;
        target.imageSmoothingEnabled = false;
        drawVillagerSprite(target, villager, {
          x: size / 2,
          y: size - 1,
          scale: 0.8,
          frame: 0,
          facing: 1,
          pose: 'idle',
        });
      });
      window.Hud.onSelectVillager = (id) => {
        const v = villagerById.get(id);
        if (!v) return;
        camera.follow = false;
        camera.followId = id;
        camera.centreOn(v.x, v.y);
        camera.zoom = Math.max(camera.zoom, 2.2);
        updateFollowLabel();
        openVillager(id);
      };
    }

    data.agents.forEach((agent) => {
      const v = villagerById.get(agent.id);
      if (v) {
        v.status = agent.status;
        v.task = agent.task;
        v.lastOutput = agent.lastOutput;
        v.progress = agent.progress;
      }
      window.Hud.updateAgent(agent);
    });

    window.Hud.updateStats(data.stats);
    window.Hud.renderListings(data.listings);
    shorts = data.shorts || [];
    window.Hud.renderShorts(shorts);
    refreshActiveBuildings();
  }

  function refreshActiveBuildings() {
    activeBuildings.clear();
    villagers.forEach((v) => {
      if (v.status === 'working') activeBuildings.add(v.home);
    });
  }

  /* ======================================================================
     Controls
     ====================================================================== */

  function updateFollowLabel() {
    document.getElementById('follow-state').textContent = camera.follow ? 'on' : 'off';
  }

  document.getElementById('btn-follow').addEventListener('click', () => {
    camera.follow = !camera.follow;
    window.Sfx.select();
    updateFollowLabel();
  });

  document.getElementById('btn-mute').addEventListener('click', (event) => {
    const muted = window.Sfx.toggle();
    event.currentTarget.textContent = muted ? '🔇' : '🔊';
  });

  document.getElementById('btn-run').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    window.Sfx.confirm();
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 1 }),
      });
      const data = await response.json();
      if (!response.ok || data.accepted === false) {
        window.Hud.toast(data.reason || 'The village is busy.', 'warn');
      }
    } catch (error) {
      window.Hud.toast('Could not start a run: ' + error.message, 'error');
    } finally {
      setTimeout(() => {
        button.disabled = false;
      }, 1500);
    }
  });

  /**
   * The cinema panel's buttons. Approve/reject/reroll mirror the Telegram card;
   * "metrics" records a view count so the treasury re-estimates the payout.
   */
  window.Hud.onShortAction = async (action, shortId, payload) => {
    try {
      const options = { method: 'POST' };
      if (action === 'metrics') {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(payload || {});
      }
      const response = await fetch(`/api/shorts/${shortId}/${action}`, options);
      const data = await response.json();

      if (!response.ok || data.accepted === false || data.ok === false) {
        window.Hud.toast(data.reason || data.detail || `Could not ${action} #${shortId}.`, 'warn');
        return;
      }
      window.Hud.toast(`Short #${shortId}: ${action} sent.`, 'success');
      refreshShorts();
    } catch (error) {
      window.Hud.toast('Request failed: ' + error.message, 'error');
    }
  };

  window.Hud.onGenerateShort = async () => {
    try {
      const response = await fetch('/api/shorts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await response.json();
      if (!response.ok || data.accepted === false) {
        window.Hud.toast(data.reason || 'The village is busy.', 'warn');
        return;
      }
      window.Hud.toast('The Bard is writing a new Short…', 'info');
    } catch (error) {
      window.Hud.toast('Could not start the Bard: ' + error.message, 'error');
    }
  };

  /** Re-pull the Shorts list after an action that changed one. */
  async function refreshShorts() {
    try {
      const response = await fetch('/api/shorts');
      const data = await response.json();
      shorts = data.shorts || [];
      window.Hud.renderShorts(shorts);
    } catch {
      // The socket will push the next update anyway.
    }
  }

  /* -------------------------------------------------------------- deals */

  /** Approve/reject a curated deal, mirroring the Telegram card. */
  window.Hud.onDealAction = async (action, dealId) => {
    try {
      const response = await fetch(`/api/deals/${dealId}/${action}`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        window.Hud.toast(data.detail || `Could not ${action} deal #${dealId}.`, 'warn');
        return;
      }
      window.Hud.toast(`Deal #${dealId}: ${action} sent.`, 'success');
      refreshDeals();
    } catch (error) {
      window.Hud.toast('Request failed: ' + error.message, 'error');
    }
  };

  /** Record clicks/earnings copied from the Associates dashboard. */
  window.Hud.onDealMetrics = async (dealId, payload) => {
    try {
      const response = await fetch(`/api/deals/${dealId}/metrics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {}),
      });
      const data = await response.json();
      if (!response.ok) {
        window.Hud.toast(data.detail || 'Could not record metrics.', 'warn');
        return;
      }
      window.Hud.toast(`Deal #${dealId}: metrics saved.`, 'success');
      refreshDeals();
    } catch (error) {
      window.Hud.toast('Request failed: ' + error.message, 'error');
    }
  };

  /** Re-pull the deals list. Also the initial load for the panel. */
  async function refreshDeals() {
    try {
      const response = await fetch('/api/deals');
      const data = await response.json();
      window.Hud.renderDeals(data);
    } catch {
      // Non-fatal: the panel simply keeps whatever it last rendered.
    }
  }

  window.Hud.onScoutDeal = async () => {
    // There is no dashboard-side generate endpoint for deals: curation runs
    // through the Scout on the CLI or /scout in Telegram, both of which post
    // the card for approval. Say so rather than failing silently.
    window.Hud.toast('Run  python main.py --scout  or send /scout in Telegram.', 'info');
  };

  refreshDeals();

  window.Hud.onListingAction = async (action, listingId) => {
    try {
      const response = await fetch(`/api/listings/${listingId}/${action}`, { method: 'POST' });
      const data = await response.json();
      if (data.ok === false && data.accepted === undefined) {
        window.Hud.toast(`Could not ${action} #${listingId}.`, 'warn');
      } else {
        window.Hud.toast(`Listing #${listingId}: ${action} sent.`, 'success');
      }
    } catch (error) {
      window.Hud.toast('Request failed: ' + error.message, 'error');
    }
  };

  window.addEventListener('resize', resize);

  /* ======================================================================
     Boot
     ====================================================================== */

  bakeTextures();
  buildMap();
  resize();
  updateFollowLabel();
  connect();
  requestAnimationFrame(frame);

  // Exposed for debugging from the console.
  window.Village = { camera, villagers: () => villagers, BUILDINGS, findPath, tiles };
})();

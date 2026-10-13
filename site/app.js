(() => {
  'use strict';
  const content = window.AESCODE_CONTENT || {gallery: [], comparisons: []};
  const gallery = content.gallery;
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const create = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  let expanded = false;
  let highlightedId = null;
  const pageSize = 18;
  function highlightCard(id) {
    highlightedId = id;
    $$('.gallery-card').forEach(card => {
      const active = card.dataset.galleryId === id;
      card.classList.toggle('is-peeked', active);
      card.querySelector('.gallery-image').setAttribute('aria-pressed', String(active));
    });
  }
  function positionHighlight(container, image, region, overlay) {
    if (!region || !image.naturalWidth || !image.clientWidth || !image.clientHeight) { overlay.hidden = true; return; }
    const containerRect = container.getBoundingClientRect();
    const imageRect = image.getBoundingClientRect();
    const ratio = image.naturalWidth / image.naturalHeight;
    const width = Math.min(imageRect.width, imageRect.height * ratio);
    const height = width / ratio;
    overlay.style.left = `${imageRect.left - containerRect.left - container.clientLeft + (imageRect.width - width) / 2 + region[0] * width}px`;
    overlay.style.top = `${imageRect.top - containerRect.top - container.clientTop + (imageRect.height - height) / 2 + region[1] * height}px`;
    overlay.style.width = `${region[2] * width}px`;
    overlay.style.height = `${region[3] * height}px`;
    overlay.hidden = false;
  }

  const narrowGallery = window.matchMedia('(max-width: 800px)');
  function galleryCard(item, ordinal = gallery.indexOf(item) + 1) {
    const article = create('article', 'gallery-card');
    article.dataset.galleryId = item.id;
    article.classList.toggle('is-peeked', highlightedId === item.id);
    const preview = create('button', 'gallery-image');
    preview.type = 'button';
    preview.setAttribute('aria-label', `Highlight ${item.title}`);
    preview.setAttribute('aria-pressed', String(highlightedId === item.id));
    preview.addEventListener('click', () => highlightCard(highlightedId === item.id ? null : item.id));
    const image = create('img');
    Object.assign(image, {src: item.src, alt: item.alt, loading: 'lazy', decoding: 'async'});
    if (item.width && item.height) { image.width = item.width; image.height = item.height; }
    preview.append(image);
    const heading = create('div', 'gallery-card-heading');
    heading.append(create('h3', '', item.title), create('span', '', String(ordinal).padStart(2, '0')));
    article.append(preview, heading);
    return article;
  }
  function renderGallery() {
    const base = gallery.slice(0, pageSize);
    const extra = !expanded && gallery.find(item => item.id === content.galleryPreviewExtraId && item.height > item.width && !base.some(featured => featured.id === item.id));
    const shown = expanded ? gallery : extra ? [...base, extra] : base;
    if (!shown.some(item => item.id === highlightedId)) highlightedId = null;
    const grid = $('#gallery-grid');
    grid.replaceChildren();
    grid.classList.toggle('gallery-grid-with-corner', Boolean(extra));
    if (extra) {
      const count = narrowGallery.matches ? 2 : 3;
      const perColumn = Math.ceil(base.length / count);
      const groups = Array.from({length: count}, (_, index) => base.slice(index * perColumn, (index + 1) * perColumn));
      groups[narrowGallery.matches ? count - 1 : 0].push(extra);
      for (const items of groups) {
        const column = create('div', 'gallery-preview-column');
        for (const item of items) column.append(galleryCard(item, item.id === extra.id ? base.length + 1 : gallery.indexOf(item) + 1));
        grid.append(column);
      }
    } else {
      for (const item of shown) grid.append(galleryCard(item));
    }
    $('#gallery-count').textContent = `${shown.length} / ${gallery.length} EXAMPLES`;
    const more = $('#show-more');
    more.hidden = gallery.length <= pageSize;
    more.querySelector('.expand-label').textContent = expanded ? 'Show fewer' : `Explore all ${gallery.length} designs`;
    more.querySelector('.expand-arrow').textContent = expanded ? '↑' : '↓';
    more.setAttribute('aria-expanded', String(expanded));
    more.setAttribute('aria-controls', 'gallery-grid');
  }
  narrowGallery.addEventListener('change', () => { if (!expanded) renderGallery(); });
  $('#show-more').addEventListener('click', () => {
    const collapsing = expanded;
    expanded = !expanded;
    renderGallery();
    if (collapsing) $('#gallery-grid').scrollIntoView({block: 'start'});
  });
  $('#gallery').addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); highlightCard(null); }
  });
  document.addEventListener('click', event => {
    if (highlightedId && !event.target.closest('.gallery-image')) highlightCard(null);
  });
  renderGallery();
  if (!gallery.length) $('#gallery-count').textContent = 'Showcase data unavailable. See the examples in the repository.';

  const heroChoices = ['case-9272bf68ffe1bb', 'case-f3c6b3975bf539'].map(id => gallery.find(item => item.id === id));
  $$('[data-hero]').forEach((figure, index) => {
    const item = heroChoices[index] || gallery.find(item => index ? item.height > item.width : item.width > item.height);
    if (!item) return;
    Object.assign(figure.querySelector('img'), {src: item.src, alt: item.alt});
  });

  // Exact reported values from the repository README, not computed or rounded estimates.
  const results = [
    ['GPT-5.5', false, 88.67, 86.18, 82.85, 85.90, 79.27, 68.09, 52.80, 66.72, 76.31],
    ['GPT-5.5', true, 89.22, 86.85, 81.35, 85.80, 82.44, 89.93, 57.91, 76.76, 81.28],
    ['Claude Opus 4.8', false, 91.40, 85.03, 87.69, 88.04, 84.98, 64.88, 46.72, 65.53, 76.78],
    ['Claude Opus 4.8', true, 92.45, 84.43, 73.75, 83.55, 86.42, 89.76, 55.51, 77.23, 80.39],
    ['Qwen3-VL-8B', false, 58.32, 77.23, 43.30, 59.61, 38.33, 23.97, 13.22, 25.17, 42.39],
    ['Qwen3-VL-8B', true, 64.07, 70.99, 42.35, 59.14, 52.90, 51.33, 29.92, 44.72, 51.93],
    ['Qwen3-VL-32B', false, 66.00, 80.31, 41.36, 62.55, 49.81, 34.85, 19.34, 34.67, 48.61],
    ['Qwen3-VL-32B', true, 71.19, 79.61, 50.77, 67.19, 62.04, 64.59, 38.43, 55.02, 61.10],
    ['AesCode-8B', true, 94.06, 88.36, 87.79, 90.07, 86.41, 87.80, 53.21, 75.80, 82.94],
    ['AesCode-32B', true, 95.34, 97.27, 90.37, 94.33, 85.76, 90.58, 55.99, 77.44, 85.89]
  ];
  const metrics = {overall: {column: 10, description: 'Overall = mean of Rule and Visual. Higher is better.'}, visual: {column: 9, description: 'Visual = mean of Content, Layout, and Style. Higher is better.'}, rule: {column: 5, description: 'Rule = mean of Text, Bound, and Chart. Higher is better.'}};
  const chartRows = [];
  for (const row of results.filter(row => row[1])) {
    const ours = row[0].startsWith('AesCode');
    const element = create('div', `chart-row${ours ? ' ours' : ''}`);
    const label = create('span', 'chart-row-label', row[0]);
    const track = create('div', 'bar-track');
    track.setAttribute('aria-hidden', 'true');
    const fill = create('div', 'bar-fill');
    track.append(fill);
    const value = create('span', 'chart-row-value');
    element.append(label, track, value);
    $('#results-chart').append(element);
    chartRows.push({row, fill, value});
  }
  function setMetric(key) {
    const metric = metrics[key];
    for (const {row, fill, value} of chartRows) {
      fill.style.setProperty('--score', `${row[metric.column]}%`);
      value.textContent = row[metric.column].toFixed(2);
    }
    $$('[data-metric]').forEach(button => {
      const selected = button.dataset.metric === key;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    $('#metric-description').textContent = metric.description;
  }
  $$('[data-metric]').forEach(button => button.addEventListener('click', () => setMetric(button.dataset.metric)));
  setMetric('overall');

  let comparisonHighlightRefresh = () => {};
  const comparisonObserver = new ResizeObserver(() => comparisonHighlightRefresh());
  function setComparison(index) {
    const item = content.comparisons[index];
    const pair = $('#comparison-pair');
    comparisonObserver.disconnect();
    comparisonHighlightRefresh = () => {};
    pair.replaceChildren();
    const images = [{title: `${item.title} — ${item.otherModel}`, src: item.otherSrc, alt: `${item.otherModel}: ${item.alt}`, highlight: item.otherHighlight}, {title: `${item.title} — AesCode-32B`, src: item.oursSrc, alt: `AesCode-32B: ${item.alt}`}];
    images.forEach((image, i) => {
      const panel = create('article', `comparison-panel${i ? ' ours' : ''}`);
      const header = create('header');
      header.append(create('span', '', i ? 'AesCode-32B' : item.otherModel), create('span', '', i ? 'SAME PROMPT' : 'ISSUE OUTLINED IN RED'));
      const preview = create('div', 'comparison-image');
      const img = create('img');
      Object.assign(img, {src: image.src, alt: image.alt, loading: 'lazy', decoding: 'async'});
      preview.append(img);
      if (image.highlight) {
        const overlay = create('span', 'comparison-highlight');
        overlay.setAttribute('aria-hidden', 'true');
        overlay.hidden = true;
        overlay.append(create('span', '', '!'));
        preview.append(overlay);
        comparisonHighlightRefresh = () => positionHighlight(preview, img, image.highlight, overlay);
        img.addEventListener('load', comparisonHighlightRefresh, {once: true});
        comparisonObserver.observe(preview);
        requestAnimationFrame(comparisonHighlightRefresh);
      }
      const note = create('p', `comparison-focus-note${i ? ' resolved' : ''}`, i ? item.oursNote : item.focusLabel);
      panel.append(header, preview, note);
      pair.append(panel);
    });
    $('#comparison-description').textContent = item.description;
    $$('#comparison-tabs button').forEach((button, i) => {
      button.classList.toggle('active', i === index);
      button.setAttribute('aria-pressed', String(i === index));
    });
  }
  content.comparisons.forEach((item, index) => {
    const button = create('button', 'filter', item.title);
    button.addEventListener('click', () => setComparison(index));
    $('#comparison-tabs').append(button);
  });
  if (content.comparisons.length) setComparison(0);
  else $('#comparisons').hidden = true;

  const film = $('#project-film');
  $$('[data-time]').forEach(button => button.addEventListener('click', async () => {
    const time = Number(button.dataset.time);
    $('#film-status').textContent = '';
    film.scrollIntoView({block: 'center'});
    try {
      if (film.readyState < 1) {
        await new Promise((resolve, reject) => {
          const clean = () => { film.removeEventListener('loadedmetadata', loaded); film.removeEventListener('error', failed); clearTimeout(timer); };
          const loaded = () => { clean(); resolve(); };
          const failed = () => { clean(); reject(new Error('Video could not be loaded.')); };
          const timer = setTimeout(failed, 20000);
          film.addEventListener('loadedmetadata', loaded, {once: true});
          film.addEventListener('error', failed, {once: true});
          film.load();
        });
      }
      film.currentTime = time;
      await film.play();
    } catch {
      $('#film-status').textContent = 'Use the video play control to start playback, or download the film using the link below.';
    }
  }));
  film.addEventListener('error', () => {
    $('#film-status').textContent = 'The film could not be played. Try again, or download the film using the link below.';
  });

  const trainingFigure = $('#training-dynamics');
  const motionToggle = $('#training-motion-toggle');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let sweepRequested = !reducedMotion.matches;
  let trainingVisible = false;
  const updateTrainingMotion = () => {
    const enabled = sweepRequested && !reducedMotion.matches;
    trainingFigure.classList.toggle('is-sweeping', enabled && trainingVisible && !document.hidden);
    trainingFigure.classList.toggle('motion-reduced', reducedMotion.matches);
    motionToggle.disabled = reducedMotion.matches;
    motionToggle.setAttribute('aria-pressed', String(enabled));
    motionToggle.textContent = reducedMotion.matches ? 'Motion off' : enabled ? 'Pause sweep' : 'Play sweep';
    motionToggle.setAttribute('aria-label', reducedMotion.matches ? 'Animation disabled by reduced-motion preference' : enabled ? 'Pause decorative training sweep' : 'Play decorative training sweep');
  };
  motionToggle.addEventListener('click', () => { sweepRequested = !sweepRequested; updateTrainingMotion(); });
  reducedMotion.addEventListener('change', () => { sweepRequested = !reducedMotion.matches; updateTrainingMotion(); });
  document.addEventListener('visibilitychange', updateTrainingMotion);
  new IntersectionObserver(entries => {
    trainingVisible = entries[0].isIntersecting;
    updateTrainingMotion();
  }, {threshold: 0.1}).observe(trainingFigure);
  updateTrainingMotion();
})();

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initAnalyticsModeControls();
    initAnalyticsSplitter();
    loadStats();
    initMap();
    initForm();
    loadCities();
    initTrendForecastControls();
    initFeaturePages();
    initPreferenceSliders();
    initRentAnalyticsControls();
    refreshPlannerData();
});

let latestPredictionInput = null;
let latestPredictedRent = null;
let latestPredictionResult = null;
let latestCostBreakdown = null;
let latestRecommendations = [];
let latestBudgetAdvisor = [];
let latestMarketInsights = null;
let latestTrendForecast = null;
let trendForecastChart = null;
let trendsPageChart = null;
let recommendationRefreshTimer = null;
let costRefreshTimer = null;
let map;
let baseMapTileLayer = null;
let mapMarkerIndex = new Map();
let currentMapPoints = [];
let mapHeatLayer = null;
let mapZoneLayer = null;
let mapMarkerLayer = null;
let mapFocusMarker = null;

const DEFAULT_COST_INPUTS = {
    deposit_months: 2,
    brokerage_months: 1,
    maintenance: 2500,
    utilities: 3000,
    parking: 1500,
    moving_cost: 8000
};

const WEIGHT_PRESETS = {
    balanced: { cost_weight: 35, commute_weight: 15, safety_weight: 20, transit_weight: 15, amenity_weight: 15 },
    budget: { cost_weight: 55, commute_weight: 10, safety_weight: 15, transit_weight: 10, amenity_weight: 10 },
    commute: { cost_weight: 20, commute_weight: 40, safety_weight: 15, transit_weight: 15, amenity_weight: 10 },
    family: { cost_weight: 20, commute_weight: 10, safety_weight: 30, transit_weight: 10, amenity_weight: 30 }
};

const SAVED_SEARCH_INPUT_IDS = ['analytics-search-name', 'saved-search-name'];
const THEME_STORAGE_KEY = 'urbanmove-theme';
const ANALYTICS_MODE_STORAGE_KEY = 'urbanmove-analytics-mode';
const ANALYTICS_SPLIT_STORAGE_KEY = 'urbanmove-analytics-split';
const WORK_MAP_STATUS_DEFAULT = 'Paste a pinned map link and the app will read latitude and longitude automatically.';
const FALLBACK_API_BASES = ['http://127.0.0.1:5000', 'http://localhost:5000'];
const DEFAULT_ANALYTICS_SPLITS = { classic: 42, glass: 58 };
const MAP_LIVABILITY_BANDS = [
    { key: 'needs-work', label: 'Low intensity', min: 0, max: 4.4, color: '#ef4444', weight: 0.18, radius: 18 },
    { key: 'balanced', label: 'Medium intensity', min: 4.5, max: 6.4, color: '#f59e0b', weight: 0.42, radius: 24 },
    { key: 'strong-fit', label: 'High intensity', min: 6.5, max: 7.9, color: '#22c55e', weight: 0.7, radius: 30 },
    { key: 'premium-fit', label: 'Peak intensity', min: 8.0, max: 10.0, color: '#06b6d4', weight: 1.0, radius: 36 }
];
const MAP_RENT_BANDS = [
    { key: 'budget', label: 'Budget', max: 29999, color: '#6366f1' },
    { key: 'mid-range', label: 'Mid-range', max: 49999, color: '#f59e0b' },
    { key: 'premium', label: 'Premium', max: Infinity, color: '#ef4444' }
];
const MAP_BUDGET_FIT_BANDS = [
    { key: 'budget-safe', label: 'Comfort fit', color: '#22c55e' },
    { key: 'budget-stretch', label: 'Stretch fit', color: '#f59e0b' },
    { key: 'budget-over', label: 'Over budget', color: '#ef4444' }
];
let currentTheme = 'dark';
let resolvedApiBase = null;
let currentAnalyticsSplits = { ...DEFAULT_ANALYTICS_SPLITS };

function uniqueValues(values) {
    return [...new Set(values.filter((value) => typeof value === 'string' && value.trim()))];
}

function normalizeApiBase(base) {
    return String(base || '').trim().replace(/\/+$/, '');
}

function getApiBaseCandidates() {
    if (resolvedApiBase) return [resolvedApiBase];

    const candidates = [];
    const explicitBase = normalizeApiBase(document.body?.dataset?.apiBase || window.urbanmoveApiBase);
    if (explicitBase) {
        candidates.push(explicitBase);
    }

    const protocol = window.location?.protocol || '';
    const origin = normalizeApiBase(window.location?.origin);
    if (origin && origin !== 'null' && /^https?:$/i.test(protocol)) {
        candidates.push(origin);
    }

    FALLBACK_API_BASES.forEach((base) => candidates.push(base));
    return uniqueValues(candidates);
}

function buildApiUrl(path, base = '') {
    if (/^https?:\/\//i.test(path)) {
        return path;
    }

    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    if (!base) {
        return normalizedPath;
    }

    return new URL(normalizedPath, `${normalizeApiBase(base)}/`).toString();
}

async function apiFetch(path, options = {}) {
    const normalizedPath = /^https?:\/\//i.test(path)
        ? path
        : (path.startsWith('/') ? path : `/${path}`);

    const candidates = /^https?:\/\//i.test(path)
        ? [{ base: '', url: path }]
        : getApiBaseCandidates().map((base) => ({ base, url: buildApiUrl(normalizedPath, base) }));

    let lastNetworkError = null;
    let lastResponse = null;

    for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        try {
            const response = await fetch(candidate.url, options);
            const canRetryOnNotFound = normalizedPath.startsWith('/api/')
                && response.status === 404
                && i < candidates.length - 1;

            if (canRetryOnNotFound) {
                lastResponse = response;
                continue;
            }

            if (candidate.base && response.status < 500) {
                resolvedApiBase = candidate.base;
            }

            return response;
        } catch (err) {
            lastNetworkError = err;
        }
    }

    if (lastResponse) {
        return lastResponse;
    }

    throw new Error('Failed to fetch. Start the Flask app with "python app.py".');
}

function readStoredTheme() {
    try {
        const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
        return stored === 'light' ? 'light' : 'dark';
    } catch (_err) {
        return 'dark';
    }
}

function writeStoredTheme(theme) {
    try {
        window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (_err) {
        // Ignore storage failures and keep the session theme only.
    }
}

function readStoredAnalyticsMode() {
    try {
        const stored = window.localStorage.getItem(ANALYTICS_MODE_STORAGE_KEY);
        return stored === 'glass' ? 'glass' : 'classic';
    } catch (_err) {
        return 'classic';
    }
}

function writeStoredAnalyticsMode(mode) {
    try {
        window.localStorage.setItem(ANALYTICS_MODE_STORAGE_KEY, mode === 'glass' ? 'glass' : 'classic');
    } catch (_err) {
        // Ignore storage failures and keep the session mode only.
    }
}

function readStoredAnalyticsSplits() {
    try {
        const parsed = JSON.parse(window.localStorage.getItem(ANALYTICS_SPLIT_STORAGE_KEY) || '{}');
        return {
            classic: Number.isFinite(Number(parsed.classic)) ? Number(parsed.classic) : DEFAULT_ANALYTICS_SPLITS.classic,
            glass: Number.isFinite(Number(parsed.glass)) ? Number(parsed.glass) : DEFAULT_ANALYTICS_SPLITS.glass
        };
    } catch (_err) {
        return { ...DEFAULT_ANALYTICS_SPLITS };
    }
}

function writeStoredAnalyticsSplits(splits) {
    try {
        window.localStorage.setItem(ANALYTICS_SPLIT_STORAGE_KEY, JSON.stringify({
            classic: splits.classic,
            glass: splits.glass
        }));
    } catch (_err) {
        // Ignore storage failures and keep the session split only.
    }
}

function getCurrentTheme() {
    return currentTheme === 'light' ? 'light' : 'dark';
}

function getThemePalette() {
    const theme = getCurrentTheme();
    if (theme === 'light') {
        return {
            chartTick: '#475569',
            chartLegend: '#334155',
            chartGrid: 'rgba(148, 163, 184, 0.28)',
            tooltipBg: 'rgba(255, 255, 255, 0.98)',
            tooltipText: '#0f172a',
            tooltipBorder: '#4f46e5'
        };
    }

    return {
        chartTick: '#94a3b8',
        chartLegend: '#cbd5e1',
        chartGrid: 'rgba(148, 163, 184, 0.15)',
        tooltipBg: 'rgba(30, 41, 59, 0.95)',
        tooltipText: '#f8fafc',
        tooltipBorder: '#6366f1'
    };
}

function updateThemeToggleButton() {
    const button = document.getElementById('theme-toggle');
    if (!button) return;

    const isLight = getCurrentTheme() === 'light';
    const nextThemeLabel = isLight ? 'dark' : 'light';
    button.setAttribute('aria-pressed', String(isLight));
    button.setAttribute('aria-label', `Switch to ${nextThemeLabel} mode`);

    const text = button.querySelector('.theme-toggle-text');
    if (text) {
        text.textContent = isLight ? 'Light mode' : 'Dark mode';
    }
}

function getThemeTileUrl() {
    return getCurrentTheme() === 'light'
        ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
}

function applyChartTheme(chart) {
    if (!chart?.options) return;
    const palette = getThemePalette();
    const scales = chart.options.scales || {};

    if (chart.options.plugins?.legend?.labels) {
        chart.options.plugins.legend.labels.color = palette.chartLegend;
    }

    if (chart.options.plugins?.tooltip) {
        chart.options.plugins.tooltip.backgroundColor = palette.tooltipBg;
        chart.options.plugins.tooltip.titleColor = palette.tooltipText;
        chart.options.plugins.tooltip.bodyColor = palette.tooltipText;
        chart.options.plugins.tooltip.borderColor = palette.tooltipBorder;
    }

    if (scales.x?.ticks) {
        scales.x.ticks.color = palette.chartTick;
    }
    if (scales.x?.grid && typeof scales.x.grid.display !== 'boolean') {
        scales.x.grid.color = palette.chartGrid;
    }
    if (scales.y?.ticks) {
        scales.y.ticks.color = palette.chartTick;
    }
    if (scales.y?.grid) {
        scales.y.grid.color = palette.chartGrid;
    }

    chart.update('none');
}

function refreshThemeAwareCharts() {
    applyChartTheme(window.cityChart);
    applyChartTheme(trendForecastChart);
    applyChartTheme(trendsPageChart);
}

function refreshMapTheme() {
    if (!baseMapTileLayer) return;
    baseMapTileLayer.setUrl(getThemeTileUrl());
}

function applyTheme(theme, options = {}) {
    currentTheme = theme === 'light' ? 'light' : 'dark';
    if (document.body) {
        document.body.dataset.theme = currentTheme;
    }
    updateThemeToggleButton();
    refreshMapTheme();
    refreshThemeAwareCharts();

    if (options.persist !== false) {
        writeStoredTheme(currentTheme);
    }
}

function initThemeToggle() {
    applyTheme(readStoredTheme(), { persist: false });

    const button = document.getElementById('theme-toggle');
    if (!button) return;

    button.addEventListener('click', () => {
        applyTheme(getCurrentTheme() === 'light' ? 'dark' : 'light');
    });
}

function updateAnalyticsModeButtons(mode) {
    document.querySelectorAll('[data-analytics-mode]').forEach((button) => {
        if (button.tagName !== 'BUTTON') return;
        const isActive = button.dataset.analyticsMode === mode;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-pressed', String(isActive));
    });
}

function getCurrentAnalyticsMode() {
    const predictSection = document.getElementById('predict');
    return predictSection?.dataset?.analyticsMode === 'glass' ? 'glass' : 'classic';
}

function clampAnalyticsSplit(value, mode = getCurrentAnalyticsMode()) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return DEFAULT_ANALYTICS_SPLITS[mode] || DEFAULT_ANALYTICS_SPLITS.classic;
    }

    const limits = mode === 'glass'
        ? { min: 40, max: 74 }
        : { min: 30, max: 68 };
    return Math.max(limits.min, Math.min(limits.max, numeric));
}

function applyAnalyticsSplit(value, options = {}) {
    const predictSection = document.getElementById('predict');
    if (!predictSection) return;

    const mode = options.mode === 'glass' ? 'glass' : (options.mode === 'classic' ? 'classic' : getCurrentAnalyticsMode());
    const clamped = clampAnalyticsSplit(value, mode);
    predictSection.style.setProperty('--analytics-form-basis', `${clamped}%`);
    predictSection.style.setProperty('--analytics-result-basis', `${100 - clamped}%`);
    currentAnalyticsSplits[mode] = clamped;

    if (options.persist !== false) {
        writeStoredAnalyticsSplits(currentAnalyticsSplits);
    }
}

function updateAnalyticsModeStatus(mode = getCurrentAnalyticsMode()) {
    const status = document.getElementById('analytics-mode-status');
    if (!status) return;

    const ratio = currentAnalyticsSplits[mode] || DEFAULT_ANALYTICS_SPLITS[mode] || DEFAULT_ANALYTICS_SPLITS.classic;
    if (mode === 'glass') {
        status.innerHTML = `
            <span class="analytics-status-pill analytics-status-pill-glass">Glass active</span>
            <strong>Dark pop-up cards are enabled.</strong>
            <span>Large step cards lead the page and the divider is set to ${Math.round(ratio)}% form width.</span>
        `;
        return;
    }

    status.innerHTML = `
        <span class="analytics-status-pill analytics-status-pill-classic">Classic active</span>
        <strong>Standard workspace with a resize handle.</strong>
        <span>Drag the divider between input and prediction. Current form width: ${Math.round(ratio)}%.</span>
    `;
}

function applyAnalyticsMode(mode, options = {}) {
    const predictSection = document.getElementById('predict');
    if (!predictSection) return;

    const normalizedMode = mode === 'glass' ? 'glass' : 'classic';
    predictSection.dataset.analyticsMode = normalizedMode;
    updateAnalyticsModeButtons(normalizedMode);
    applyAnalyticsSplit(currentAnalyticsSplits[normalizedMode] || DEFAULT_ANALYTICS_SPLITS[normalizedMode], {
        mode: normalizedMode,
        persist: options.persist
    });
    updateAnalyticsModeStatus(normalizedMode);

    if (options.persist !== false) {
        writeStoredAnalyticsMode(normalizedMode);
    }
}

function initAnalyticsModeControls() {
    currentAnalyticsSplits = readStoredAnalyticsSplits();
    applyAnalyticsMode(readStoredAnalyticsMode(), { persist: false });

    document.querySelectorAll('button[data-analytics-mode]').forEach((button) => {
        button.addEventListener('click', () => applyAnalyticsMode(button.dataset.analyticsMode));
    });
}

function initAnalyticsSplitter() {
    const splitter = document.getElementById('analytics-splitter');
    const layout = document.querySelector('#predict .predict-layout');
    const predictSection = document.getElementById('predict');
    if (!splitter || !layout || !predictSection) return;

    let dragging = false;

    const updateFromPointer = (clientX) => {
        const rect = layout.getBoundingClientRect();
        if (!rect.width) return;
        const ratio = ((clientX - rect.left) / rect.width) * 100;
        applyAnalyticsSplit(ratio);
        updateAnalyticsModeStatus();
    };

    const stopDragging = () => {
        if (!dragging) return;
        dragging = false;
        layout.classList.remove('is-resizing');
        document.body.classList.remove('analytics-resizing');
    };

    splitter.addEventListener('pointerdown', (event) => {
        if (window.matchMedia('(max-width: 960px)').matches) return;
        dragging = true;
        layout.classList.add('is-resizing');
        document.body.classList.add('analytics-resizing');
        splitter.setPointerCapture?.(event.pointerId);
        updateFromPointer(event.clientX);
        event.preventDefault();
    });

    splitter.addEventListener('pointermove', (event) => {
        if (!dragging) return;
        updateFromPointer(event.clientX);
    });

    splitter.addEventListener('pointerup', () => stopDragging());
    splitter.addEventListener('pointercancel', () => stopDragging());
    splitter.addEventListener('lostpointercapture', () => stopDragging());

    splitter.addEventListener('keydown', (event) => {
        const mode = getCurrentAnalyticsMode();
        const step = event.shiftKey ? 4 : 2;
        const current = currentAnalyticsSplits[mode] || DEFAULT_ANALYTICS_SPLITS[mode];

        if (event.key === 'ArrowLeft') {
            applyAnalyticsSplit(current - step, { mode });
            updateAnalyticsModeStatus(mode);
            event.preventDefault();
        }

        if (event.key === 'ArrowRight') {
            applyAnalyticsSplit(current + step, { mode });
            updateAnalyticsModeStatus(mode);
            event.preventDefault();
        }

        if (event.key === 'Home') {
            applyAnalyticsSplit(DEFAULT_ANALYTICS_SPLITS[mode], { mode });
            updateAnalyticsModeStatus(mode);
            event.preventDefault();
        }
    });
}

function setPreferenceFlowStatus(message = '') {
    const status = document.getElementById('preferences-flow-status');
    if (status) status.textContent = message;
}

// Load cities and localities dynamically
async function loadCities() {
    try {
        const res = await apiFetch('/api/cities');
        const data = await res.json();
        
        // Update city dropdown
        const citySelect = document.getElementById('city');
        if (citySelect && data.cities) {
            // Clear existing options except the first one (or keep all)
            const currentValue = citySelect.value;
            citySelect.innerHTML = '';
            data.cities.forEach(city => {
                const option = document.createElement('option');
                option.value = city;
                option.textContent = city;
                if (city === currentValue) option.selected = true;
                citySelect.appendChild(option);
            });
            citySelect.addEventListener('change', () => {
                const workCity = document.getElementById('work_city');
                if (workCity && !workCity.value) {
                    workCity.value = citySelect.value;
                }
            });
        }

        // Prefill city text inputs across feature pages for convenience.
        const defaultCity = citySelect?.value || ((data.cities && data.cities.length > 0) ? data.cities[0] : 'Mumbai');
        ['trnd-city', 'score-city', 'work_city']
            .forEach((id) => {
                const el = document.getElementById(id);
                if (el && !el.value) {
                    el.value = defaultCity;
                }
            });
        
        // Update locality input with autocomplete suggestions (optional enhancement)
        // For now, we'll just store the data for potential future use
        window.citiesData = data;
        refreshSavedSearchNameSuggestions();
    } catch (err) {
        console.error("Error loading cities:", err);
    }
}

// Navigation
function showSection(id) {
    if (typeof event !== 'undefined' && event && typeof event.preventDefault === 'function') {
        event.preventDefault();
    }
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    document.getElementById(id).classList.add('active');
    if (typeof event !== 'undefined' && event && event.currentTarget?.classList?.contains('nav-item')) {
        event.currentTarget.classList.add('active');
    } else {
        const navItem = document.querySelector(`.nav-item[onclick*="${id}"]`);
        if (navItem) navItem.classList.add('active');
    }

    if (id === 'map-view') {
        setTimeout(() => map.invalidateSize(), 100);
    }
    if (id === 'trnd-view') {
        const summary = document.getElementById('trnd-summary');
        if (summary && !safeText(summary.textContent).trim()) {
            loadTrendsPage();
        }
    }
    if (id === 'comp-view') {
        ensureCompareHelperText();
    }
    if (id === 'score-view') {
        const result = document.getElementById('score-result');
        if (result && !safeText(result.textContent).trim()) {
            loadLocalityScorecard();
        }
    }
    if (id === 'planner-view') {
        refreshPlannerData();
    }
}

function buildMapLocationKey(lat, lon) {
    const numericLat = Number(lat);
    const numericLon = Number(lon);
    if (!Number.isFinite(numericLat) || !Number.isFinite(numericLon)) return '';
    return `${numericLat.toFixed(5)},${numericLon.toFixed(5)}`;
}

function getMapLivabilityBand(score) {
    const numericScore = Number(score);
    const safeScore = Number.isFinite(numericScore) ? numericScore : 0;
    return MAP_LIVABILITY_BANDS.find((band, index) => (
        safeScore >= band.min && (safeScore <= band.max || index === MAP_LIVABILITY_BANDS.length - 1)
    )) || MAP_LIVABILITY_BANDS[0];
}

function getMapRentBand(rent) {
    const numericRent = Number(rent);
    const safeRent = Number.isFinite(numericRent) ? numericRent : 0;
    return MAP_RENT_BANDS.find((band) => safeRent <= band.max) || MAP_RENT_BANDS[MAP_RENT_BANDS.length - 1];
}

function getLiveBudgetTargetValue() {
    const numeric = parseNumberOrNull(document.getElementById('budget_target')?.value);
    return numeric !== null && numeric > 0 ? numeric : null;
}

function getMapBudgetFitBand(rent, budgetTarget) {
    const safeRent = Number(rent) || 0;
    const safeBudget = Number(budgetTarget) || 0;
    if (!safeBudget) return MAP_BUDGET_FIT_BANDS[1];

    if (safeRent <= safeBudget * 0.9) {
        return MAP_BUDGET_FIT_BANDS[0];
    }
    if (safeRent <= safeBudget * 1.1) {
        return MAP_BUDGET_FIT_BANDS[1];
    }
    return MAP_BUDGET_FIT_BANDS[2];
}

function getActiveMapMarkerBand(rent, budgetTarget = getLiveBudgetTargetValue()) {
    return budgetTarget ? getMapBudgetFitBand(rent, budgetTarget) : getMapRentBand(rent);
}

function formatBudgetGap(rent, budgetTarget) {
    const safeRent = Number(rent) || 0;
    const safeBudget = Number(budgetTarget) || 0;
    if (!safeBudget) return '';

    const gap = safeRent - safeBudget;
    if (Math.abs(gap) <= safeBudget * 0.03) {
        return 'Near your budget target';
    }
    if (gap < 0) {
        return `${formatCurrency(Math.abs(gap))} under target`;
    }
    return `${formatCurrency(gap)} above target`;
}

function formatMapLivabilityRange(band) {
    return `${band.min.toFixed(1)} - ${band.max.toFixed(1)} / 10`;
}

function formatMapRentRange(band, index) {
    if (index === 0) {
        return `Up to ${formatCurrency(band.max)}`;
    }

    const start = MAP_RENT_BANDS[index - 1].max + 1;
    if (!Number.isFinite(band.max)) {
        return `${formatCurrency(start)} and above`;
    }

    return `${formatCurrency(start)} - ${formatCurrency(band.max)}`;
}

function renderMapCategoryLegend(points = [], budgetTarget = getLiveBudgetTargetValue()) {
    const panel = document.getElementById('map-category-legend');
    if (!panel) return;

    const totalPoints = Array.isArray(points) ? points.length : 0;
    if (totalPoints === 0) {
        panel.innerHTML = `
            <div class="map-info-header">
                <div>
                    <h3>How the map is categorized</h3>
                    <p class="section-subtitle">Heat zones will appear here after the map data loads.</p>
                </div>
            </div>
        `;
        return;
    }

    const livabilityCounts = MAP_LIVABILITY_BANDS.map((band) => ({ ...band, count: 0 }));
    const markerBands = (budgetTarget ? MAP_BUDGET_FIT_BANDS : MAP_RENT_BANDS).map((band) => ({ ...band, count: 0 }));

    points.forEach((point) => {
        const livabilityBand = getMapLivabilityBand(point.Neighborhood_Livability_Score);
        const markerBand = getActiveMapMarkerBand(point.Rent, budgetTarget);
        const livabilityTarget = livabilityCounts.find((band) => band.key === livabilityBand.key);
        const markerTarget = markerBands.find((band) => band.key === markerBand.key);
        if (livabilityTarget) livabilityTarget.count += 1;
        if (markerTarget) markerTarget.count += 1;
    });

    panel.innerHTML = `
        <div class="map-info-header">
            <div>
                <h3>How the map is categorized</h3>
                <p class="section-subtitle">Glow intensity increases with livability. Marker colors switch to budget-fit mode when a budget target is set in Rent Analytics.</p>
            </div>
            <div class="map-info-summary-stack">
                <div class="map-info-summary">${totalPoints} mapped locations</div>
                <div class="map-info-summary ${budgetTarget ? 'is-budget-active' : 'is-budget-waiting'}">
                    ${budgetTarget ? `Budget target ${formatCurrency(budgetTarget)} active` : 'Set a budget to enable affordability colors'}
                </div>
            </div>
        </div>
        <div class="map-legend-grid">
            <section class="map-legend-section">
                <div class="map-legend-section-head">
                    <h4>Heat Intensity</h4>
                    <p>Higher livability creates stronger glow and a larger colored zone.</p>
                </div>
                ${livabilityCounts.map((band) => `
                    <div class="map-legend-item">
                        <span class="map-legend-swatch map-tone-${band.key}"></span>
                        <div class="map-legend-copy">
                            <strong>${escapeHtml(band.label)}</strong>
                            <span>${escapeHtml(formatMapLivabilityRange(band))}</span>
                        </div>
                        <span class="map-legend-count">${band.count} areas</span>
                    </div>
                `).join('')}
            </section>
            <section class="map-legend-section">
                <div class="map-legend-section-head">
                    <h4>${budgetTarget ? 'Budget Fit Markers' : 'Rent Range Markers'}</h4>
                    <p>${budgetTarget ? 'Dots compare each listing rent against the current budget target.' : 'Dots show the rent band of each listing point until a budget target is set.'}</p>
                </div>
                ${markerBands.map((band, index) => `
                    <div class="map-legend-item">
                        <span class="map-legend-swatch map-tone-${band.key}"></span>
                        <div class="map-legend-copy">
                            <strong>${escapeHtml(band.label)}</strong>
                            <span>${escapeHtml(budgetTarget
                                ? (band.key === 'budget-safe'
                                    ? 'At least 10% below the current budget'
                                    : band.key === 'budget-stretch'
                                        ? 'Within about 10% of the current budget'
                                        : 'More than 10% above the current budget')
                                : formatMapRentRange(band, index))}</span>
                        </div>
                        <span class="map-legend-count">${band.count} listings</span>
                    </div>
                `).join('')}
            </section>
        </div>
        <div class="map-legend-note">Selecting a location from recommendations jumps this heat map to that exact point first. The external Google Maps link is now secondary.</div>
    `;
}

function buildMapPopupContent(point, livabilityBand, markerBand, budgetTarget = getLiveBudgetTargetValue()) {
    const locationName = safeText(point['Area Locality']) || safeText(point.City) || 'Unknown Location';
    const city = safeText(point.City) || 'Unknown city';
    const rent = Number(point.Rent) || 0;
    const livabilityScore = parseNumberOrNull(point.Neighborhood_Livability_Score);
    const budgetGap = formatBudgetGap(rent, budgetTarget);
    const metaItems = [
        point.BHK ? `<span>${escapeHtml(safeText(point.BHK))} BHK</span>` : '',
        point.Size ? `<span>${escapeHtml(safeText(point.Size))} sqft</span>` : '',
        point['Furnishing Status'] ? `<span>${escapeHtml(safeText(point['Furnishing Status']))}</span>` : '',
        point['Area Type'] ? `<span>${escapeHtml(safeText(point['Area Type']))}</span>` : ''
    ].filter(Boolean).join('');
    const gmapsUrl = `https://www.google.com/maps?q=${Number(point.Latitude)},${Number(point.Longitude)}`;

    return `
        <div class="map-popup-card">
            <div class="map-popup-head">
                <div>
                    <div class="map-popup-title">${escapeHtml(locationName)}</div>
                    <div class="map-popup-subtitle">${escapeHtml(city)}</div>
                </div>
                <div class="map-popup-rent">${formatCurrency(rent)}</div>
            </div>
            <div class="map-popup-badges">
                <span class="map-popup-badge map-tone-${markerBand.key}">${escapeHtml(markerBand.label)}${budgetTarget ? '' : ' rent'}</span>
                <span class="map-popup-badge map-tone-${livabilityBand.key}">${escapeHtml(livabilityBand.label)} livability</span>
                ${budgetGap ? `<span class="map-popup-badge">${escapeHtml(budgetGap)}</span>` : ''}
                ${point.Cluster_ID !== undefined && point.Cluster_ID !== null ? `<span class="map-popup-badge">Cluster ${escapeHtml(safeText(point.Cluster_ID))}</span>` : ''}
            </div>
            ${metaItems ? `<div class="map-popup-meta">${metaItems}</div>` : ''}
            <div class="map-popup-grid">
                <div class="map-popup-stat">
                    <span>Livability</span>
                    <strong>${livabilityScore !== null ? `${livabilityScore.toFixed(1)} / 10` : 'Not rated'}</strong>
                </div>
                <div class="map-popup-stat">
                    <span>Coordinates</span>
                    <strong>${Number(point.Latitude).toFixed(4)}, ${Number(point.Longitude).toFixed(4)}</strong>
                </div>
            </div>
            <a class="map-popup-link" href="${gmapsUrl}" target="_blank" rel="noopener noreferrer">Open exact location in Google Maps</a>
        </div>
    `;
}

function clearMapFocusMarker() {
    if (!map || !mapFocusMarker) return;
    map.removeLayer(mapFocusMarker);
    mapFocusMarker = null;
}

function renderMapVisualization(points = currentMapPoints) {
    if (!map) return;

    currentMapPoints = Array.isArray(points) ? points : [];
    const budgetTarget = getLiveBudgetTargetValue();
    renderMapCategoryLegend(currentMapPoints, budgetTarget);
    mapMarkerIndex = new Map();
    clearMapFocusMarker();

    if (mapHeatLayer) {
        map.removeLayer(mapHeatLayer);
        mapHeatLayer = null;
    }
    if (mapZoneLayer) {
        map.removeLayer(mapZoneLayer);
        mapZoneLayer = null;
    }
    if (mapMarkerLayer) {
        map.removeLayer(mapMarkerLayer);
        mapMarkerLayer = null;
    }

    if (!currentMapPoints.length) return;

    mapZoneLayer = L.layerGroup().addTo(map);
    mapMarkerLayer = L.layerGroup().addTo(map);

    const heatGradient = MAP_LIVABILITY_BANDS.reduce((accumulator, band, index) => {
        accumulator[index === 0 ? 0.1 : band.weight] = band.color;
        return accumulator;
    }, {});

    if (L.heatLayer) {
        const heatPoints = currentMapPoints.map((point) => {
            const numericScore = Number(point.Neighborhood_Livability_Score);
            const normalizedScore = Number.isFinite(numericScore) ? Math.max(0.14, Math.min(1, numericScore / 10)) : 0.18;
            return [Number(point.Latitude), Number(point.Longitude), normalizedScore];
        });

        mapHeatLayer = L.heatLayer(heatPoints, {
            radius: 34,
            blur: 26,
            maxZoom: 17,
            minOpacity: 0.32,
            gradient: heatGradient,
            pane: 'heatmapPane'
        }).addTo(map);
    }

    currentMapPoints.forEach((point) => {
        const livabilityBand = getMapLivabilityBand(point.Neighborhood_Livability_Score);
        const markerBand = getActiveMapMarkerBand(point.Rent, budgetTarget);
        const lat = Number(point.Latitude);
        const lon = Number(point.Longitude);
        const locationName = safeText(point['Area Locality']) || safeText(point.City) || 'Unknown Location';
        const rent = Number(point.Rent) || 0;
        const numericScore = Number(point.Neighborhood_Livability_Score);
        const zoneOpacity = Number.isFinite(numericScore) ? Math.max(0.16, Math.min(0.3, numericScore / 36)) : 0.16;

        L.circleMarker([lat, lon], {
            radius: livabilityBand.radius,
            fillColor: livabilityBand.color,
            color: livabilityBand.color,
            weight: 1,
            fillOpacity: zoneOpacity,
            opacity: 0.34,
            pane: 'zonePane',
            interactive: false
        }).addTo(mapZoneLayer);

        const marker = L.circleMarker([lat, lon], {
            radius: budgetTarget ? 7 : 6,
            fillColor: markerBand.color,
            color: '#ffffff',
            weight: 1.8,
            fillOpacity: 0.92,
            className: 'marker-point',
            pane: 'markerPane',
            zIndexOffset: 500
        });

        marker.bindPopup(buildMapPopupContent(point, livabilityBand, markerBand, budgetTarget));
        marker.bindTooltip(
            `<div class="map-tooltip-label"><strong>${escapeHtml(locationName)}</strong><span>${formatCurrency(rent)}</span></div>`,
            {
                permanent: false,
                direction: 'top',
                offset: [0, -10],
                className: 'map-label'
            }
        );

        marker.on('click', (event) => {
            if (event.originalEvent) {
                event.originalEvent.preventDefault();
                event.originalEvent.stopPropagation();
            }
            marker.openPopup();
        });

        marker.addTo(mapMarkerLayer);
        if (marker.bringToFront) marker.bringToFront();
        mapMarkerIndex.set(buildMapLocationKey(lat, lon), marker);
    });
}

function refreshMapVisualization() {
    if (!map || !currentMapPoints.length) return;
    renderMapVisualization(currentMapPoints);
}

function focusMapLocation(lat, lon, options = {}) {
    const numericLat = Number(lat);
    const numericLon = Number(lon);
    if (!Number.isFinite(numericLat) || !Number.isFinite(numericLon) || !map) return;

    const zoom = Number.isFinite(Number(options.zoom)) ? Number(options.zoom) : 13;
    const locationKey = buildMapLocationKey(numericLat, numericLon);
    showSection('map-view');

    window.setTimeout(() => {
        map.invalidateSize();
        if (typeof map.flyTo === 'function') {
            map.flyTo([numericLat, numericLon], zoom, { duration: 0.8 });
        } else {
            map.setView([numericLat, numericLon], zoom);
        }

        const marker = mapMarkerIndex.get(locationKey);
        if (marker) {
            clearMapFocusMarker();
            window.setTimeout(() => marker.openPopup(), 240);
            return;
        }

        clearMapFocusMarker();
        mapFocusMarker = L.circleMarker([numericLat, numericLon], {
            radius: 11,
            fillColor: '#ffffff',
            color: '#38bdf8',
            weight: 3,
            fillOpacity: 0.22,
            pane: 'markerPane'
        }).addTo(map);
        mapFocusMarker.bindPopup(`
            <div class="map-popup-card">
                <div class="map-popup-head">
                    <div>
                        <div class="map-popup-title">Focused recommendation location</div>
                        <div class="map-popup-subtitle">This point came from the recommendation card.</div>
                    </div>
                </div>
                <div class="map-popup-grid">
                    <div class="map-popup-stat">
                        <span>Coordinates</span>
                        <strong>${numericLat.toFixed(4)}, ${numericLon.toFixed(4)}</strong>
                    </div>
                </div>
                <a class="map-popup-link" href="https://www.google.com/maps?q=${numericLat},${numericLon}" target="_blank" rel="noopener noreferrer">Open exact location in Google Maps</a>
            </div>
        `);
        window.setTimeout(() => mapFocusMarker?.openPopup(), 240);
    }, 120);
}

// Stats
async function loadStats() {
    try {
        const res = await apiFetch('/api/stats');
        const data = await res.json();
        const avgRentEl = document.getElementById('avg-rent-stat');
        const overallAvgRent = Math.round(data.avg_rent);
        avgRentEl.textContent = `Rs ${overallAvgRent.toLocaleString()}`;
        document.getElementById('total-listings-stat').textContent = data.total_listings.toLocaleString();
        document.getElementById('cities-count-stat').textContent = data.total_cities || data.cities.length;
        
        // Show rent range if available
        if (data.min_rent && data.max_rent) {
            const rentRangeEl = document.getElementById('rent-range-stat');
            if (rentRangeEl) {
                rentRangeEl.textContent = `Rs ${Math.round(data.min_rent/1000)}k - Rs ${Math.round(data.max_rent/1000)}k`;
                rentRangeEl.style.fontSize = '0.8rem';
                rentRangeEl.style.color = 'var(--text-muted)';
                rentRangeEl.style.marginTop = '0.5rem';
            }
        }
        
        // Show median rent if available
        if (data.med_rent) {
            const medRentEl = document.getElementById('med-rent-stat');
            if (medRentEl) {
                medRentEl.textContent = `Rs ${Math.round(data.med_rent).toLocaleString()}`;
            }
        }

        // Init Chart with REAL data
        const ctx = document.getElementById('cityChart');
        if (ctx) {
            // Check if we have rent_by_city data
            const labels = Object.keys(data.rent_by_city || {});
            const values = Object.values(data.rent_by_city || {});

            // Fallback if empty (though stats file has it)
            const chartData = labels.length > 0 ? values : data.cities.map(() => 0);
            const chartLabels = labels.length > 0 ? labels : data.cities;

            // Store chart instance globally for potential updates
            window.cityChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: chartLabels,
                    datasets: [{
                        label: 'Avg Rent (Rs)',
                        data: chartData,
                        backgroundColor: 'rgba(99, 102, 241, 0.8)',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2,
                    onHover: (_event, activeEls) => {
                        if (!avgRentEl) return;
                        if (!activeEls || activeEls.length === 0) {
                            avgRentEl.textContent = `Rs ${overallAvgRent.toLocaleString()}`;
                            return;
                        }
                        const point = activeEls[0];
                        const city = chartLabels[point.index];
                        const rent = chartData[point.index];
                        if (city && typeof rent === 'number') {
                            avgRentEl.textContent = `${city}: Rs ${Math.round(rent).toLocaleString()}`;
                        }
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    layout: {
                        padding: {
                            top: 10,
                            bottom: 10,
                            left: 10,
                            right: 18
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: { 
                            callbacks: { 
                                title: (items) => items?.[0]?.label || '',
                                label: (c) => `Avg Rent: Rs ${Math.round(c.raw).toLocaleString()}` 
                            },
                            backgroundColor: 'rgba(30, 41, 59, 0.95)',
                            titleColor: '#f8fafc',
                            bodyColor: '#f8fafc',
                            borderColor: '#6366f1',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { 
                                color: '#94a3b8',
                                callback: (val) => 'Rs ' + Math.round(val / 1000) + 'k',
                                maxTicksLimit: 8
                            }
                        },
                        x: { 
                            grid: { display: false }, 
                            ticks: { 
                                color: '#94a3b8',
                                maxRotation: 45,
                                minRotation: 0
                            } 
                        }
                    }
                }
            });
            applyChartTheme(window.cityChart);
        }
    } catch (err) {
        console.error("Stats error", err);
        document.getElementById('avg-rent-stat').textContent = "Error";
    }
}

// Map
async function initMap() {
    map = L.map('map').setView([22.3511, 78.6677], 5);
    mapMarkerIndex = new Map();
    currentMapPoints = [];

    map.on('click', (e) => {
        const { lat, lng } = e.latlng;
        const gmapsUrl = `https://www.google.com/maps?q=${lat},${lng}`;
        window.open(gmapsUrl, '_blank', 'noopener');
    });

    map.createPane('heatmapPane');
    map.getPane('heatmapPane').style.zIndex = 300;
    map.createPane('zonePane');
    map.getPane('zonePane').style.zIndex = 340;
    map.getPane('zonePane').style.pointerEvents = 'none';
    baseMapTileLayer = L.tileLayer(getThemeTileUrl(), {
        attribution: '(c) CARTO'
    });
    baseMapTileLayer.addTo(map);
    renderMapCategoryLegend();

    try {
        const res = await apiFetch('/api/map_data');
        const data = await res.json();
        currentMapPoints = Array.isArray(data) ? data.filter((point) => (
            Number.isFinite(Number(point.Latitude)) && Number.isFinite(Number(point.Longitude))
        )) : [];
        renderMapVisualization(currentMapPoints);
    } catch (err) {
        console.error("Map data error", err);
        renderMapCategoryLegend();
    }
}

// Predict
function formatCurrency(value) {
    const numeric = Number(value);
    const safeValue = Number.isFinite(numeric) ? numeric : 0;
    return `Rs ${Math.round(safeValue).toLocaleString('en-IN')}`;
}

function escapeHtml(value) {
    return safeText(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function parseNumberOrNull(value) {
    if (value === null || value === undefined || value === '') return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function isValidCoordinatePair(lat, lon) {
    return Number.isFinite(lat) && Number.isFinite(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}

function parseCoordinatePair(rawLat, rawLon) {
    const lat = Number(rawLat);
    const lon = Number(rawLon);
    return isValidCoordinatePair(lat, lon) ? { lat, lon } : null;
}

function extractCoordinatesFromMapUrl(rawValue) {
    const text = safeText(rawValue).trim();
    if (!text) return null;

    let decoded = text;
    try {
        decoded = decodeURIComponent(text);
    } catch (_err) {
        decoded = text;
    }

    if (/^-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?$/.test(decoded)) {
        const [lat, lon] = decoded.split(',').map((value) => value.trim());
        return parseCoordinatePair(lat, lon);
    }

    const patterns = [
        /@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/,
        /!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)/,
        /#map=\d+\/(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)/,
        /[?&#](?:q|query|ll|center|destination)=(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/
    ];

    for (const pattern of patterns) {
        const match = decoded.match(pattern);
        if (!match) continue;
        const parsed = parseCoordinatePair(match[1], match[2]);
        if (parsed) return parsed;
    }

    return null;
}

function setWorkCoordinates(lat, lon) {
    const latInput = document.getElementById('work_lat');
    const lonInput = document.getElementById('work_lon');
    if (latInput) latInput.value = lat === null || lat === undefined ? '' : String(lat);
    if (lonInput) lonInput.value = lon === null || lon === undefined ? '' : String(lon);
}

function setWorkMapStatus(message = WORK_MAP_STATUS_DEFAULT) {
    const status = document.getElementById('work-map-status');
    if (status) status.textContent = message;
}

function buildMapUrlFromCoordinates(lat, lon) {
    const numericLat = Number(lat);
    const numericLon = Number(lon);
    if (!isValidCoordinatePair(numericLat, numericLon)) return '';
    return `https://www.google.com/maps?q=${numericLat},${numericLon}`;
}

function syncWorkLocationFromMapUrl(options = {}) {
    const { showStatus = false } = options;
    const input = document.getElementById('work_map_url');
    if (!input) {
        return { work_map_url: '', work_lat: null, work_lon: null };
    }

    const workMapUrl = safeText(input.value).trim();
    if (!workMapUrl) {
        setWorkCoordinates(null, null);
        if (showStatus) setWorkMapStatus();
        return { work_map_url: '', work_lat: null, work_lon: null };
    }

    const coordinates = extractCoordinatesFromMapUrl(workMapUrl);
    if (!coordinates) {
        setWorkCoordinates(null, null);
        if (showStatus) {
            setWorkMapStatus('No coordinates found in that link yet. Paste a pinned map URL with a visible location.');
        }
        return { work_map_url: workMapUrl, work_lat: null, work_lon: null };
    }

    setWorkCoordinates(coordinates.lat, coordinates.lon);
    if (showStatus) {
        setWorkMapStatus(`Location detected: ${coordinates.lat.toFixed(4)}, ${coordinates.lon.toFixed(4)}`);
    }

    return {
        work_map_url: workMapUrl,
        work_lat: coordinates.lat,
        work_lon: coordinates.lon
    };
}

function getTrustLabel(score) {
    const numeric = Number(score);
    if (!Number.isFinite(numeric)) return 'Trust pending';
    if (numeric >= 80) return `High trust ${numeric.toFixed(0)}`;
    if (numeric >= 60) return `Watch ${numeric.toFixed(0)}`;
    return `Low trust ${numeric.toFixed(0)}`;
}

function getTrustClass(score) {
    const numeric = Number(score);
    if (!Number.isFinite(numeric)) return 'trust-neutral';
    if (numeric >= 80) return 'trust-high';
    if (numeric >= 60) return 'trust-mid';
    return 'trust-low';
}

function collectPredictFormInput() {
    return {
        city: safeText(document.getElementById('city')?.value).trim(),
        budget_target: parseNumberOrNull(document.getElementById('budget_target')?.value),
        locality: safeText(document.getElementById('locality')?.value).trim(),
        bhk: parseInt(document.getElementById('bhk')?.value || '1', 10),
        size: parseFloat(document.getElementById('size')?.value || '500'),
        bathroom: parseInt(document.getElementById('bathroom')?.value || '1', 10),
        area_type: safeText(document.getElementById('area_type')?.value).trim(),
        furnishing: safeText(document.getElementById('furnishing')?.value).trim(),
        tenant: safeText(document.getElementById('tenant')?.value).trim(),
        bathroom_type: safeText(document.getElementById('bathroom_type')?.value).trim()
    };
}

function collectCostAssumptions(predictedRent) {
    return {
        rent: predictedRent,
        deposit_months: parseNumberOrNull(document.getElementById('deposit_months')?.value),
        brokerage_months: parseNumberOrNull(document.getElementById('brokerage_months')?.value),
        maintenance: parseNumberOrNull(document.getElementById('maintenance')?.value),
        utilities: parseNumberOrNull(document.getElementById('utilities')?.value),
        parking: parseNumberOrNull(document.getElementById('parking')?.value),
        moving_cost: parseNumberOrNull(document.getElementById('moving_cost')?.value)
    };
}

function collectPlannerPreferences() {
    const workLocation = syncWorkLocationFromMapUrl();
    return {
        work_city: safeText(document.getElementById('work_city')?.value).trim(),
        work_map_url: workLocation.work_map_url,
        work_lat: workLocation.work_lat,
        work_lon: workLocation.work_lon,
        tenant: safeText(document.getElementById('tenant')?.value).trim(),
        has_pet: safeText(document.getElementById('has_pet')?.value).trim() === 'true',
        cost_weight: parseNumberOrNull(document.getElementById('cost_weight')?.value) ?? 35,
        commute_weight: parseNumberOrNull(document.getElementById('commute_weight')?.value) ?? 15,
        safety_weight: parseNumberOrNull(document.getElementById('safety_weight')?.value) ?? 20,
        transit_weight: parseNumberOrNull(document.getElementById('transit_weight')?.value) ?? 15,
        amenity_weight: parseNumberOrNull(document.getElementById('amenity_weight')?.value) ?? 15
    };
}

function buildCurrentSearchPayload() {
    return {
        prediction_inputs: collectPredictFormInput(),
        planner_preferences: collectPlannerPreferences(),
        cost_assumptions: {
            deposit_months: parseNumberOrNull(document.getElementById('deposit_months')?.value),
            brokerage_months: parseNumberOrNull(document.getElementById('brokerage_months')?.value),
            maintenance: parseNumberOrNull(document.getElementById('maintenance')?.value),
            utilities: parseNumberOrNull(document.getElementById('utilities')?.value),
            parking: parseNumberOrNull(document.getElementById('parking')?.value),
            moving_cost: parseNumberOrNull(document.getElementById('moving_cost')?.value)
        },
        latest_predicted_rent: latestPredictedRent
    };
}

function getSavedSearchNameInputs() {
    return SAVED_SEARCH_INPUT_IDS
        .map((id) => document.getElementById(id))
        .filter(Boolean);
}

function buildSuggestedSearchName() {
    const source = latestPredictionInput || collectPredictFormInput();
    const city = safeText(source.city).trim() || getDefaultCityValue();
    const locality = safeText(source.locality).trim();
    const bhk = parseInt(source.bhk, 10);
    const budget = getBudgetTargetValue(source);
    const parts = [];

    if (city) parts.push(city);
    if (Number.isFinite(bhk) && bhk > 0) parts.push(`${bhk} BHK`);
    if (locality) parts.push(locality);
    if (budget !== null) parts.push(`Budget ${formatCurrency(budget)}`);

    return parts.join(' | ') || 'Current rent plan';
}

function refreshSavedSearchNameSuggestions() {
    const placeholder = buildSuggestedSearchName();
    getSavedSearchNameInputs().forEach((input) => {
        input.placeholder = placeholder;
    });
}

function syncSavedSearchNameInputs(sourceId) {
    const source = document.getElementById(sourceId);
    if (!source) return;
    SAVED_SEARCH_INPUT_IDS.forEach((id) => {
        if (id === sourceId) return;
        const input = document.getElementById(id);
        if (input) input.value = source.value;
    });
}

function setSavedSearchNameValue(value = '') {
    getSavedSearchNameInputs().forEach((input) => {
        input.value = value;
    });
}

function getActiveSavedSearchName() {
    const typedName = getSavedSearchNameInputs()
        .map((input) => safeText(input.value).trim())
        .find(Boolean);
    return typedName || buildSuggestedSearchName();
}

function setSavedSearchStatus(message = '') {
    ['analytics-search-status', 'saved-search-status'].forEach((id) => {
        const status = document.getElementById(id);
        if (status) status.textContent = message;
    });
}

function syncCompareSelectionFromRecommendations(force = false) {
    const input = document.getElementById('compare-ids');
    if (!input) return;

    if (!force && safeText(input.value).trim()) {
        return;
    }

    const ids = latestRecommendations
        .map((item) => parseInt(item.sample_listing_id, 10))
        .filter((value) => Number.isFinite(value))
        .slice(0, 3);

    input.value = ids.join(',');
    if (ids.length > 0) {
        ensureCompareHelperText('Top matched homes are ready to compare.', true);
        return;
    }

    if (force) {
        ensureCompareHelperText(undefined, true);
    }
}

function syncLinkedSectionsFromCurrentPlan(options = {}) {
    const { forceCompare = false } = options;
    const source = latestPredictionInput || collectPredictFormInput();
    const city = source.city || document.getElementById('city')?.value || getDefaultCityValue();

    setInputValue('trnd-city', city);
    setInputValue('trnd-bhk', source.bhk || 2);
    setInputValue('trnd-locality', source.locality || '');
    setInputValue('score-city', city);
    setInputValue('score-bhk', source.bhk || 2);
    syncCompareSelectionFromRecommendations(forceCompare);
}

function updateAnalyticsActionBox() {
    const box = document.getElementById('analytics-actions-box');
    if (!box) return;

    refreshSavedSearchNameSuggestions();
    const hasPrediction = Number.isFinite(Number(latestPredictedRent));
    box.style.display = hasPrediction ? 'block' : 'none';
    if (!hasPrediction) {
        const status = document.getElementById('analytics-search-status');
        if (status) status.textContent = '';
    }
}

function getActivePredictionInput() {
    const baseInput = latestPredictionInput ? { ...latestPredictionInput } : collectPredictFormInput();
    baseInput.budget_target = parseNumberOrNull(document.getElementById('budget_target')?.value);
    return baseInput;
}

function initPreferenceSliders() {
    ['cost_weight', 'commute_weight', 'safety_weight', 'transit_weight', 'amenity_weight'].forEach((id) => {
        const input = document.getElementById(id);
        const output = document.getElementById(`${id}_value`);
        if (!input || !output) return;
        const sync = () => {
            output.textContent = safeText(input.value);
        };
        input.addEventListener('input', sync);
        sync();
    });
}

function initTrendForecastControls() {
    const horizonSelect = document.getElementById('trend-horizon');
    if (!horizonSelect) return;

    horizonSelect.addEventListener('change', async () => {
        if (!latestPredictionInput || !latestPredictedRent) return;
        await loadRentTrendForecast(latestPredictionInput, latestPredictedRent);
    });
}

function getBudgetTargetValue(source = latestPredictionInput) {
    const numeric = parseNumberOrNull(source?.budget_target);
    return numeric !== null && numeric > 0 ? numeric : null;
}

function resetCostInputs() {
    Object.entries(DEFAULT_COST_INPUTS).forEach(([key, value]) => setInputValue(key, value));
    scheduleCostRefresh();
}

function applyWeightPreset(presetName) {
    const preset = WEIGHT_PRESETS[presetName];
    if (!preset) return;
    Object.entries(preset).forEach(([key, value]) => setInputValue(key, value));
    scheduleRecommendationRefresh();
}

function scheduleRecommendationRefresh() {
    if (!latestPredictedRent) return;
    clearTimeout(recommendationRefreshTimer);
    latestPredictionInput = getActivePredictionInput();
    renderDecisionOverview();
    recommendationRefreshTimer = setTimeout(() => {
        const activeInput = getActivePredictionInput();
        latestPredictionInput = activeInput;
        Promise.all([
            loadRecommendations(activeInput, latestPredictedRent),
            loadBudgetAdvisor(activeInput, latestPredictedRent)
        ]).catch((err) => console.error('Recommendation refresh failed:', err));
    }, 180);
}

function scheduleCostRefresh() {
    if (!latestPredictedRent) return;
    clearTimeout(costRefreshTimer);
    costRefreshTimer = setTimeout(() => {
        loadCostBreakdown(latestPredictedRent);
    }, 160);
}

function ensureCompareHelperText(message = 'Use quick buttons from recommendations or shortlist to compare homes.', force = false) {
    const summary = document.getElementById('compare-summary');
    if (!summary) return;
    if (force || !safeText(summary.textContent).trim()) {
        summary.textContent = message;
    }
}

function addListingIdToCompare(listingId, navigateToCompare = false) {
    const numericId = parseInt(safeText(listingId), 10);
    const input = document.getElementById('compare-ids');
    if (!input || !Number.isFinite(numericId)) return;

    const ids = safeText(input.value)
        .split(',')
        .map((value) => parseInt(value.trim(), 10))
        .filter((value) => Number.isFinite(value));
    if (!ids.includes(numericId)) {
        ids.push(numericId);
    }

    input.value = ids.slice(0, 6).join(',');
    ensureCompareHelperText(`Selected listing IDs: ${input.value}`, true);

    if (navigateToCompare) {
        showSection('comp-view');
        loadCompareListings();
    }
}

function fillCompareFromRecommendations() {
    const ids = latestRecommendations
        .map((item) => parseInt(item.sample_listing_id, 10))
        .filter((value) => Number.isFinite(value))
        .slice(0, 3);
    const input = document.getElementById('compare-ids');
    if (!input) return;

    if (ids.length === 0) {
        ensureCompareHelperText('Run Rent Analytics first to compare the top matched homes.', true);
        return;
    }

    input.value = ids.join(',');
    showSection('comp-view');
    loadCompareListings();
}

async function fillCompareFromShortlist() {
    const input = document.getElementById('compare-ids');
    if (!input) return;

    try {
        const res = await apiFetch('/api/shortlist');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load shortlist');
        const ids = (data.items || [])
            .map((item) => parseInt(item.listing_id, 10))
            .filter((value) => Number.isFinite(value))
            .slice(0, 6);
        if (ids.length === 0) {
            ensureCompareHelperText('No shortlist listing IDs available yet. Save a recommendation first.', true);
            return;
        }
        input.value = ids.join(',');
        showSection('comp-view');
        loadCompareListings();
    } catch (err) {
        ensureCompareHelperText(`Error: ${err.message}`, true);
    }
}

function useCurrentSearchForTrends() {
    syncLinkedSectionsFromCurrentPlan();
    showSection('trnd-view');
    loadTrendsPage();
}

function useCurrentSearchForScorecard() {
    syncLinkedSectionsFromCurrentPlan();
    showSection('score-view');
    loadLocalityScorecard();
}

function openCompareFromAnalytics() {
    syncLinkedSectionsFromCurrentPlan({ forceCompare: true });
    const compareInput = document.getElementById('compare-ids');
    showSection('comp-view');
    if (safeText(compareInput?.value).trim()) {
        loadCompareListings();
        return;
    }
    ensureCompareHelperText('Compare homes will appear here after your matched listings are ready.', true);
}

function openPlannerFromAnalytics() {
    syncSavedSearchNameInputs('analytics-search-name');
    showSection('planner-view');
}

function initRentAnalyticsControls() {
    const resetCostsBtn = document.getElementById('reset-costs-btn');
    if (resetCostsBtn) {
        resetCostsBtn.addEventListener('click', () => resetCostInputs());
    }

    const advancedPanel = document.getElementById('advanced-options-panel');
    const preferencesOpenBtn = document.getElementById('preferences-open-btn');
    const preferencesSkipBtn = document.getElementById('preferences-skip-btn');
    const predictSubmitBtn = document.getElementById('predict-submit-btn');

    if (preferencesOpenBtn && advancedPanel) {
        preferencesOpenBtn.addEventListener('click', () => {
            advancedPanel.open = true;
            setPreferenceFlowStatus('Optional preferences are open. Review only the cards you need and skip the rest.');
            document.getElementById('locality')?.focus();
        });
    }

    if (preferencesSkipBtn && advancedPanel) {
        preferencesSkipBtn.addEventListener('click', () => {
            advancedPanel.open = false;
            setPreferenceFlowStatus('Using the default assumptions. You can predict now and adjust later only if needed.');
            predictSubmitBtn?.focus();
        });
    }

    if (advancedPanel) {
        advancedPanel.addEventListener('toggle', () => {
            if (advancedPanel.open) {
                setPreferenceFlowStatus('Optional preferences are open. Review only the cards you need and skip the rest.');
                return;
            }
            setPreferenceFlowStatus('Optional preferences are skipped for now. The default assumptions are active.');
        });
    }

    const resetAnalyticsBtn = document.getElementById('analytics-reset-btn');
    if (resetAnalyticsBtn) {
        resetAnalyticsBtn.addEventListener('click', () => resetAnalyticsForm());
    }

    ['analytics-search-name', 'saved-search-name'].forEach((id) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('input', () => syncSavedSearchNameInputs(id));
    });

    ['city', 'budget_target', 'bhk', 'locality'].forEach((id) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('input', () => refreshSavedSearchNameSuggestions());
        input.addEventListener('change', () => refreshSavedSearchNameSuggestions());
    });

    document.querySelectorAll('.preset-btn[data-preset]').forEach((button) => {
        button.addEventListener('click', () => applyWeightPreset(button.dataset.preset));
    });

    ['deposit_months', 'brokerage_months', 'maintenance', 'utilities', 'parking', 'moving_cost'].forEach((id) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('input', () => scheduleCostRefresh());
    });

    ['budget_target', 'has_pet', 'work_city', 'tenant'].forEach((id) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('input', () => {
            scheduleRecommendationRefresh();
            if (id === 'budget_target') refreshMapVisualization();
        });
        input.addEventListener('change', () => {
            scheduleRecommendationRefresh();
            if (id === 'budget_target') refreshMapVisualization();
        });
    });

    const workMapUrlInput = document.getElementById('work_map_url');
    if (workMapUrlInput) {
        workMapUrlInput.addEventListener('input', () => {
            const hasValue = safeText(workMapUrlInput.value).trim().length > 0;
            if (!hasValue) {
                setWorkCoordinates(null, null);
                setWorkMapStatus();
                scheduleRecommendationRefresh();
            }
        });
        workMapUrlInput.addEventListener('change', () => {
            syncWorkLocationFromMapUrl({ showStatus: true });
            scheduleRecommendationRefresh();
        });
        workMapUrlInput.addEventListener('blur', () => syncWorkLocationFromMapUrl({ showStatus: true }));
    }

    ['cost_weight', 'commute_weight', 'safety_weight', 'transit_weight', 'amenity_weight'].forEach((id) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('change', () => scheduleRecommendationRefresh());
    });

    ensureCompareHelperText(undefined, true);
    refreshSavedSearchNameSuggestions();
    updateAnalyticsActionBox();
    setWorkMapStatus();
    setPreferenceFlowStatus('Optional preferences are skipped for now. The default assumptions are active.');
}

function getDefaultCityValue() {
    const citySelect = document.getElementById('city');
    if (!citySelect) return 'Mumbai';
    const options = Array.from(citySelect.options || []);
    if (options.some((option) => option.value === 'Mumbai')) {
        return 'Mumbai';
    }
    return options[0]?.value || 'Mumbai';
}

function clearAnalyticsPanels() {
    latestPredictionInput = null;
    latestPredictedRent = null;
    latestPredictionResult = null;
    latestCostBreakdown = null;
    latestRecommendations = [];
    latestBudgetAdvisor = [];
    latestMarketInsights = null;
    latestTrendForecast = null;

    clearTimeout(recommendationRefreshTimer);
    clearTimeout(costRefreshTimer);

    if (trendForecastChart) {
        trendForecastChart.destroy();
        trendForecastChart = null;
    }

    const panelIds = [
        'decision-overview-box',
        'analytics-actions-box',
        'result-box',
        'cost-breakdown-box',
        'recommendations-box',
        'secondary-insights-box',
        'budget-advisor-box',
        'market-insights-box',
        'trend-forecast-box'
    ];
    panelIds.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.style.display = 'none';
        if (el.tagName === 'DETAILS') {
            el.open = false;
        }
    });

    const clearTargets = {
        'decision-overview-content': '',
        'cost-breakdown-content': '',
        'recommendations-list': '',
        'budget-advisor-list': '',
        'market-insights-content': '',
        'trend-forecast-summary': '',
        'lab-confidence': ''
    };
    Object.entries(clearTargets).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = value;
    });

    const resultValue = document.getElementById('prediction-result');
    if (resultValue) resultValue.textContent = 'Rs 0';

    const disclaimer = document.querySelector('#result-box .disclaimer');
    if (disclaimer) {
        disclaimer.textContent = 'Use this as the negotiation reference for the selected configuration.';
    }

    syncCompareSelectionFromRecommendations(true);
    updateAnalyticsActionBox();
}

function resetAnalyticsForm() {
    clearAnalyticsPanels();

    const defaults = {
        city: getDefaultCityValue(),
        budget_target: 35000,
        bhk: 2,
        tenant: 'Bachelors/Family',
        work_city: '',
        locality: '',
        size: 850,
        bathroom: 2,
        area_type: 'Super Area',
        furnishing: 'Semi-Furnished',
        bathroom_type: 'Standard',
        deposit_months: 2,
        brokerage_months: 1,
        maintenance: 2500,
        utilities: 3000,
        parking: 1500,
        moving_cost: 8000,
        has_pet: 'false',
        work_map_url: '',
        work_lat: '',
        work_lon: '',
        cost_weight: 35,
        commute_weight: 15,
        safety_weight: 20,
        transit_weight: 15,
        amenity_weight: 15
    };

    Object.entries(defaults).forEach(([id, value]) => setInputValue(id, value));
    setSavedSearchNameValue('');
    setSavedSearchStatus('');

    const advancedPanel = document.getElementById('advanced-options-panel');
    if (advancedPanel) advancedPanel.open = false;
    setPreferenceFlowStatus('Optional preferences are skipped for now. The default assumptions are active.');

    syncLinkedSectionsFromCurrentPlan({ forceCompare: true });
    refreshSavedSearchNameSuggestions();
    setWorkMapStatus();
    ensureCompareHelperText(undefined, true);
    refreshMapVisualization();
}

function renderDecisionOverview() {
    const box = document.getElementById('decision-overview-box');
    const content = document.getElementById('decision-overview-content');
    const secondaryInsightsBox = document.getElementById('secondary-insights-box');
    if (!box || !content) return;

    if (!Number.isFinite(Number(latestPredictedRent))) {
        content.innerHTML = '';
        box.style.display = 'none';
        if (secondaryInsightsBox) secondaryInsightsBox.style.display = 'none';
        updateAnalyticsActionBox();
        return;
    }

    if (secondaryInsightsBox) {
        secondaryInsightsBox.style.display = 'block';
    }

    const budgetTarget = getBudgetTargetValue();
    const topRecommendation = latestRecommendations[0] || null;
    const budgetAlternative = latestBudgetAdvisor[0] || null;
    const predictedRent = Number(latestPredictedRent);

    let budgetValue = 'No budget target';
    let budgetClass = 'trust-neutral';
    if (budgetTarget !== null) {
        const gap = predictedRent - budgetTarget;
        if (gap <= 0) {
            budgetValue = `${formatCurrency(Math.abs(gap))} under target`;
            budgetClass = 'trust-high';
        } else {
            budgetValue = `${formatCurrency(gap)} above target`;
            budgetClass = gap <= budgetTarget * 0.1 ? 'trust-mid' : 'trust-low';
        }
    }

    const summaryPoints = [];
    if (topRecommendation) {
        summaryPoints.push(
            `Best overall locality is ${escapeHtml(topRecommendation.locality)} in ${escapeHtml(topRecommendation.city)} at ${formatCurrency(topRecommendation.avg_rent)} with ${Number(topRecommendation.match_score || 0).toFixed(1)}% match.`
        );
    }
    if (budgetAlternative) {
        summaryPoints.push(
            `Closest budget-fit fallback is ${escapeHtml(budgetAlternative.locality)} at ${formatCurrency(budgetAlternative.avg_rent)}.`
        );
    }
    if (latestCostBreakdown) {
        summaryPoints.push(
            `Estimated move-in cash is ${formatCurrency(latestCostBreakdown.move_in_cash)} and monthly carry cost is ${formatCurrency(latestCostBreakdown.monthly_total)}.`
        );
    }
    if (latestMarketInsights && Number.isFinite(Number(latestMarketInsights.percentile))) {
        summaryPoints.push(
            `This estimate sits around the ${Number(latestMarketInsights.percentile).toFixed(1)}th percentile of the selected market.`
        );
    }
    if (latestTrendForecast && normalizeOptionalText(latestTrendForecast.trend_direction)) {
        const nextMonthText = Number.isFinite(Number(latestTrendForecast.next_month_forecast))
            ? ` with next-month forecast around ${formatCurrency(latestTrendForecast.next_month_forecast)}`
            : '';
        summaryPoints.push(
            `Short-term market direction is ${escapeHtml(latestTrendForecast.trend_direction)}${nextMonthText}.`
        );
    }

    content.innerHTML = `
        <div class="decision-overview-grid">
            <div class="decision-overview-stat">
                <div class="decision-overview-label">Predicted rent</div>
                <div class="decision-overview-value">${formatCurrency(predictedRent)}</div>
            </div>
            <div class="decision-overview-stat">
                <div class="decision-overview-label">Budget vs estimate</div>
                <div class="decision-overview-value">
                    <span class="trust-badge ${budgetClass}">${escapeHtml(budgetValue)}</span>
                </div>
            </div>
            <div class="decision-overview-stat">
                <div class="decision-overview-label">Best locality</div>
                <div class="decision-overview-value is-location">${escapeHtml(topRecommendation?.locality || 'Pending recommendation')}</div>
            </div>
            <div class="decision-overview-stat">
                <div class="decision-overview-label">Move-in cash</div>
                <div class="decision-overview-value">${latestCostBreakdown ? formatCurrency(latestCostBreakdown.move_in_cash) : 'Calculating...'}</div>
            </div>
        </div>
        <div class="decision-overview-summary">
            <div class="decision-overview-summary-title">Decision summary</div>
            <div class="decision-overview-list">
                ${summaryPoints.length > 0
                    ? summaryPoints.map((point) => `<div>${point}</div>`).join('')
                    : '<div>Detailed recommendation panels are still loading.</div>'}
            </div>
        </div>
    `;
    box.style.display = 'block';
    updateAnalyticsActionBox();
}

function initForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const data = collectPredictFormInput();

        const btn = form.querySelector('button[type="submit"]');
        const originalText = btn.textContent;
        btn.textContent = 'Calculating...';
        btn.disabled = true;

        try {
            const res = await apiFetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.error || 'Prediction failed');
            }

            const result = await res.json();
            const resultBox = document.getElementById('result-box');
            const predictionEl = document.getElementById('prediction-result');
            const labConfidence = document.getElementById('lab-confidence');

            resultBox.style.display = 'flex';
            predictionEl.textContent = formatCurrency(result.predicted_rent);

            if (result.location) {
                const disclaimer = resultBox.querySelector('.disclaimer');
                if (disclaimer) {
                    disclaimer.textContent = `Based on historical market data for ${result.location.city}${result.location.locality ? ` - ${result.location.locality}` : ''}`;
                }
            }

            if (labConfidence && result.confidence_range) {
                labConfidence.textContent = `Confidence range: ${formatCurrency(result.confidence_range.low)} - ${formatCurrency(result.confidence_range.high)} (expected ${formatCurrency(result.confidence_range.expected)})`;
            }

            if (result.predicted_rent) {
                latestPredictionInput = data;
                latestPredictedRent = result.predicted_rent;
                latestPredictionResult = result;
                latestCostBreakdown = null;
                latestRecommendations = [];
                latestBudgetAdvisor = [];
                latestMarketInsights = null;
                latestTrendForecast = null;
                syncLinkedSectionsFromCurrentPlan({ forceCompare: true });
                renderDecisionOverview();

                const feedbackPredicted = document.getElementById('feedback-predicted');
                if (feedbackPredicted) {
                    feedbackPredicted.value = Math.round(result.predicted_rent);
                }

                await Promise.all([
                    loadCostBreakdown(result.predicted_rent),
                    loadRecommendations(data, result.predicted_rent),
                    loadBudgetAdvisor(data, result.predicted_rent),
                    loadMarketInsights(data, result.predicted_rent),
                    loadRentTrendForecast(data, result.predicted_rent)
                ]);
            }
        } catch (err) {
            alert(`Error during prediction: ${err.message}`);
            console.error('Prediction error:', err);
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    });
}

async function loadCostBreakdown(predictedRent) {
    const box = document.getElementById('cost-breakdown-box');
    const content = document.getElementById('cost-breakdown-content');
    if (!box || !content) return;

    box.style.display = 'none';
    content.innerHTML = '';

    try {
        const res = await apiFetch('/api/cost_breakdown', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(collectCostAssumptions(predictedRent))
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Cost breakdown failed');
        latestCostBreakdown = data;

        content.innerHTML = `
            <div class="cost-summary-grid">
                <div class="cost-summary-stat">
                    <div class="cost-summary-label">Monthly total</div>
                    <div class="cost-summary-value">${formatCurrency(data.monthly_total)}</div>
                </div>
                <div class="cost-summary-stat">
                    <div class="cost-summary-label">Move-in cash</div>
                    <div class="cost-summary-value">${formatCurrency(data.move_in_cash)}</div>
                </div>
                <div class="cost-summary-stat">
                    <div class="cost-summary-label">6-month total</div>
                    <div class="cost-summary-value">${formatCurrency(data.six_month_total)}</div>
                </div>
                <div class="cost-summary-stat">
                    <div class="cost-summary-label">12-month total</div>
                    <div class="cost-summary-value">${formatCurrency(data.twelve_month_total)}</div>
                </div>
            </div>
            <div class="cost-line-list">
                <div class="cost-line-item"><span>Monthly rent</span><strong>${formatCurrency(data.line_items.monthly_rent)}</strong></div>
                <div class="cost-line-item"><span>Maintenance</span><strong>${formatCurrency(data.line_items.monthly_maintenance)}</strong></div>
                <div class="cost-line-item"><span>Utilities</span><strong>${formatCurrency(data.line_items.monthly_utilities)}</strong></div>
                <div class="cost-line-item"><span>Parking</span><strong>${formatCurrency(data.line_items.monthly_parking)}</strong></div>
                <div class="cost-line-item"><span>Deposit</span><strong>${formatCurrency(data.line_items.security_deposit)}</strong></div>
                <div class="cost-line-item"><span>Brokerage</span><strong>${formatCurrency(data.line_items.brokerage_fee)}</strong></div>
                <div class="cost-line-item"><span>Moving cost</span><strong>${formatCurrency(data.line_items.moving_cost)}</strong></div>
            </div>
        `;
        box.style.display = 'block';
        renderDecisionOverview();
    } catch (err) {
        latestCostBreakdown = null;
        content.innerHTML = `Error: ${escapeHtml(err.message)}`;
        box.style.display = 'block';
        renderDecisionOverview();
    }
}

// Load personalized locality recommendations from backend after prediction.
async function loadRecommendations(formInput, predictedRent) {
    const box = document.getElementById('recommendations-box');
    const list = document.getElementById('recommendations-list');
    if (!box || !list) return;

    list.innerHTML = '';
    box.style.display = 'none';

    try {
        const preferences = collectPlannerPreferences();
        const rankingBudget = getBudgetTargetValue(formInput) ?? predictedRent;
        const params = new URLSearchParams({
            city: formInput.city || '',
            bhk: formInput.bhk ? String(formInput.bhk) : '',
            budget: String(Math.round(rankingBudget)),
            tenant: preferences.tenant || '',
            has_pet: preferences.has_pet ? 'true' : 'false',
            work_city: preferences.work_city || formInput.city || '',
            cost_weight: String(preferences.cost_weight),
            commute_weight: String(preferences.commute_weight),
            safety_weight: String(preferences.safety_weight),
            transit_weight: String(preferences.transit_weight),
            amenity_weight: String(preferences.amenity_weight),
            limit: '5'
        });

        if (preferences.work_lat !== null) params.set('work_lat', String(preferences.work_lat));
        if (preferences.work_lon !== null) params.set('work_lon', String(preferences.work_lon));

        const res = await apiFetch(`/api/personalized_localities?${params.toString()}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Recommendation request failed');

        const recommendations = data.recommendations || [];
        latestRecommendations = recommendations;
        syncCompareSelectionFromRecommendations(true);
        renderDecisionOverview();
        if (recommendations.length === 0) {
            list.innerHTML = '<div class="mini-status">No locality matches found for the selected filters.</div>';
            box.style.display = 'block';
            return;
        }

        list.innerHTML = recommendations.map((item) => {
            const chips = (item.explanation_chips || [])
                .map((chip) => `<span class="chip">${escapeHtml(chip)}</span>`)
                .join('');
            const flagChips = []
                .concat(item.sample_trust_flags || [])
                .concat(item.sample_data_quality_flags || [])
                .slice(0, 4)
                .map((flag) => `<span class="flag-chip">${escapeHtml(flag.replace(/_/g, ' '))}</span>`)
                .join('');
            const trustLabel = getTrustLabel(item.sample_trust_score);
            const profileBadge = item.profile_source === 'curated' ? 'Curated profile' : 'Fallback profile';
            const profileNotes = normalizeOptionalText(item.profile_notes);
            const listingId = parseInt(item.sample_listing_id, 10);
            const hasCoordinates = Number.isFinite(Number(item.coordinates?.lat)) && Number.isFinite(Number(item.coordinates?.lon));
            const operationalBadges = renderListingOperationalBadges(
                item.sample_freshness_label,
                item.sample_freshness_class,
                item.sample_contact_type_label,
                item.sample_contact_type_class
            );
            const shortlistPayload = encodeURIComponent(JSON.stringify({
                listing_id: item.sample_listing_id,
                city: item.city,
                locality: item.locality,
                rent: item.avg_rent,
                bhk: formInput.bhk,
                size: formInput.size,
                notes: 'Saved from personalized locality matches'
            }));
            return `
                <div class="recommendation-card">
                    <div class="recommendation-card-header">
                        <div>
                            <div class="recommendation-title">${escapeHtml(item.locality)}</div>
                            <div class="recommendation-city">${escapeHtml(item.city)} | Match ${Number(item.match_score || 0).toFixed(1)}%</div>
                        </div>
                        <div class="recommendation-rent">${formatCurrency(item.avg_rent)}</div>
                    </div>
                    <div class="recommendation-badges">
                        <span class="trust-badge ${getTrustClass(item.sample_trust_score)}">${escapeHtml(trustLabel)}</span>
                        <span class="chip muted-chip">${escapeHtml(profileBadge)}</span>
                        ${Number.isFinite(listingId) ? `<span class="chip muted-chip">Listing #${listingId}</span>` : ''}
                        ${operationalBadges}
                    </div>
                    <div class="recommendation-meta">
                        <span>Median ${formatCurrency(item.median_rent)}</span>
                        <span>Commute ${Number(item.commute_km || 0).toFixed(1)} km</span>
                        <span>Livability ${Number(item.livability || 0).toFixed(1)}/10</span>
                        <span>Listings ${safeText(item.listing_count || 0)}</span>
                    </div>
                    <div class="score-strip">
                        <span>Cost ${Number(item.scores?.cost || 0).toFixed(0)}</span>
                        <span>Safety ${Number(item.scores?.safety || 0).toFixed(0)}</span>
                        <span>Transit ${Number(item.scores?.transit || 0).toFixed(0)}</span>
                        <span>Amenity ${Number(item.scores?.amenity || 0).toFixed(0)}</span>
                    </div>
                    ${chips ? `<div class="chip-row">${chips}</div>` : ''}
                    ${flagChips ? `<div class="chip-row">${flagChips}</div>` : ''}
                    ${profileNotes ? `<div class="recommendation-note">${escapeHtml(profileNotes)}</div>` : ''}
                    <div class="recommendation-actions">
                        ${hasCoordinates ? `<button class="btn-link focus-map-btn" type="button" data-lat="${escapeHtml(item.coordinates.lat)}" data-lon="${escapeHtml(item.coordinates.lon)}">View on heat map</button>` : ''}
                        ${Number.isFinite(listingId) ? `<button class="btn-link add-compare-btn" type="button" data-id="${listingId}">Compare</button>` : ''}
                        <button class="btn-link save-shortlist-btn" type="button" data-shortlist="${shortlistPayload}">Save to Shortlist</button>
                    </div>
                </div>
            `;
        }).join('');

        box.style.display = 'block';
    } catch (err) {
        latestRecommendations = [];
        renderDecisionOverview();
        list.innerHTML = `Error: ${escapeHtml(err.message)}`;
        box.style.display = 'block';
    }
}

// New feature: rank localities by budget fit and livability.
async function loadBudgetAdvisor(formInput, predictedRent) {
    const box = document.getElementById('budget-advisor-box');
    const list = document.getElementById('budget-advisor-list');
    if (!box || !list) return;

    list.innerHTML = '';
    box.style.display = 'none';

    try {
        const budgetTarget = getBudgetTargetValue(formInput) ?? predictedRent;
        const params = new URLSearchParams({
            city: formInput.city || '',
            bhk: formInput.bhk ? String(formInput.bhk) : '',
            budget: String(Math.round(budgetTarget)),
            limit: '5'
        });

        const res = await apiFetch(`/api/budget_advisor?${params.toString()}`);
        if (!res.ok) {
            console.error('Budget advisor fetch failed');
            return;
        }

        const data = await res.json();
        const recommendations = data.recommendations || [];
        latestBudgetAdvisor = recommendations;
        renderDecisionOverview();
        if (budgetTarget > 0 && predictedRent <= budgetTarget) {
            latestBudgetAdvisor = [];
            renderDecisionOverview();
            box.style.display = 'none';
            return;
        }
        if (recommendations.length === 0) {
            box.style.display = 'none';
            return;
        }

        recommendations.forEach((item) => {
            const card = document.createElement('div');
            card.className = 'budget-card';

            const top = document.createElement('div');
            top.className = 'budget-card-top';
            top.innerHTML = `
                <div>
                    <div class="budget-card-title">${item.locality || 'Unknown locality'}</div>
                    <div class="budget-card-subtitle">${item.city || formInput.city || 'City'}</div>
                </div>
                <div class="budget-score">Match ${item.match_score}%</div>
            `;

            const meta = document.createElement('div');
            meta.className = 'budget-meta';
            meta.innerHTML = `
                <span>Avg: ${formatCurrency(item.avg_rent)}</span>
                <span>Median: ${formatCurrency(item.median_rent)}</span>
                <span>Livability: ${Number(item.livability || 0).toFixed(1)}/10</span>
                <span>Listings: ${item.listings || 0}</span>
            `;

            const gap = document.createElement('div');
            const overBudget = Number(item.over_budget_by || 0);
            const underBudget = Number(item.savings_if_under_budget || 0);
            if (underBudget > 0) {
                gap.className = 'budget-gap-under';
                gap.textContent = `Under budget by ${formatCurrency(underBudget)}`;
            } else if (overBudget > 0) {
                gap.className = 'budget-gap-over';
                gap.textContent = `Over budget by ${formatCurrency(overBudget)}`;
            } else {
                gap.className = 'budget-gap-under';
                gap.textContent = 'Near exact budget match';
            }

            card.appendChild(top);
            card.appendChild(meta);
            card.appendChild(gap);
            list.appendChild(card);
        });

        box.style.display = 'block';
    } catch (err) {
        latestBudgetAdvisor = [];
        renderDecisionOverview();
        console.error('Error loading budget advisor:', err);
    }
}

// New feature: show where predicted rent sits in the selected market.
async function loadMarketInsights(formInput, predictedRent) {
    const box = document.getElementById('market-insights-box');
    const content = document.getElementById('market-insights-content');
    if (!box || !content) return;

    content.innerHTML = '';
    box.style.display = 'none';

    try {
        const params = new URLSearchParams({
            city: formInput.city || '',
            bhk: formInput.bhk ? String(formInput.bhk) : '',
            predicted_rent: String(Math.round(predictedRent))
        });

        const res = await apiFetch(`/api/market_insights?${params.toString()}`);
        if (!res.ok) {
            console.error('Market insights fetch failed');
            return;
        }

        const data = await res.json();
        if (!data || !data.market_size) {
            latestMarketInsights = null;
            renderDecisionOverview();
            return;
        }
        latestMarketInsights = data;

        const grid = document.createElement('div');
        grid.className = 'market-insights-grid';
        grid.innerHTML = `
            <div class="market-insight-stat">
                <div class="market-insight-label">Position</div>
                <div class="market-insight-value">${data.position_label || 'N/A'}</div>
            </div>
            <div class="market-insight-stat">
                <div class="market-insight-label">Percentile</div>
                <div class="market-insight-value">${Number(data.percentile || 0).toFixed(1)}%</div>
            </div>
            <div class="market-insight-stat">
                <div class="market-insight-label">Market Median</div>
                <div class="market-insight-value">${formatCurrency(data.median_rent)}</div>
            </div>
            <div class="market-insight-stat">
                <div class="market-insight-label">Interquartile Range</div>
                <div class="market-insight-value">${formatCurrency(data.p25_rent)} - ${formatCurrency(data.p75_rent)}</div>
            </div>
        `;

        const summary = document.createElement('div');
        summary.className = 'market-insight-summary';
        summary.innerHTML = `
            <div><strong>Compared set:</strong> ${data.market_size} listings</div>
            <div><strong>Lower than estimate:</strong> ${data.below_count}</div>
            <div><strong>Higher than estimate:</strong> ${data.above_count}</div>
            <div class="market-insight-recommendation">${data.recommendation || ''}</div>
        `;

        content.appendChild(grid);
        content.appendChild(summary);
        box.style.display = 'block';
        renderDecisionOverview();
    } catch (err) {
        latestMarketInsights = null;
        renderDecisionOverview();
        console.error('Error loading market insights:', err);
    }
}

// New feature: monthly trend + forecast chart for selected market.
async function loadRentTrendForecast(formInput, predictedRent) {
    const box = document.getElementById('trend-forecast-box');
    const summaryEl = document.getElementById('trend-forecast-summary');
    const canvas = document.getElementById('trendForecastChart');
    const horizonSelect = document.getElementById('trend-horizon');
    if (!box || !summaryEl || !canvas) return;

    box.style.display = 'none';
    summaryEl.textContent = '';

    try {
        const horizon = horizonSelect ? parseInt(horizonSelect.value, 10) : 6;
        const params = new URLSearchParams({
            city: formInput.city || '',
            bhk: formInput.bhk ? String(formInput.bhk) : '',
            locality: formInput.locality || '',
            months_history: '12',
            months_forecast: String(Number.isFinite(horizon) ? horizon : 6),
            predicted_rent: String(Math.round(predictedRent))
        });

        const res = await apiFetch(`/api/rent_trends?${params.toString()}`);
        if (!res.ok) {
            console.error('Rent trend fetch failed');
            return;
        }

        const data = await res.json();
        const historical = data.historical || [];
        const forecast = data.forecast || [];
        if (historical.length < 1) {
            latestTrendForecast = null;
            renderDecisionOverview();
            return;
        }
        latestTrendForecast = data;

        const labels = historical.map((item) => item.month).concat(forecast.map((item) => item.month));
        const historicalSeries = historical.map((item) => item.median_rent).concat(new Array(forecast.length).fill(null));

        const forecastSeries = new Array(labels.length).fill(null);
        const lastHistoricalIndex = historical.length - 1;
        forecastSeries[lastHistoricalIndex] = historical[lastHistoricalIndex].median_rent;
        forecast.forEach((item, idx) => {
            forecastSeries[historical.length + idx] = item.forecast_rent;
        });

        if (trendForecastChart) {
            trendForecastChart.destroy();
        }

        trendForecastChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Historical Median',
                        data: historicalSeries,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.15)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.28,
                        pointRadius: 2
                    },
                    {
                        label: 'Forecast',
                        data: forecastSeries,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.15)',
                        borderWidth: 2,
                        borderDash: [6, 4],
                        fill: false,
                        tension: 0.2,
                        pointRadius: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${formatCurrency(ctx.raw)}`
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#94a3b8',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 8
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.15)'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#94a3b8',
                            callback: (val) => `Rs ${Math.round(val / 1000)}k`
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.15)'
                        }
                    }
                }
            }
        });
        applyChartTheme(trendForecastChart);

        const trendLabel = data.trend_direction || 'stable';
        const avgChange = Number(data.avg_monthly_change || 0);
        const nextMonth = data.next_month_forecast;
        const isSyntheticTimeline = String(data.timeline_source || '').toLowerCase() === 'synthetic';
        let comparisonText = '';
        if (typeof data.predicted_vs_next_month === 'number') {
            const gap = data.predicted_vs_next_month;
            comparisonText = gap >= 0
                ? `Your predicted rent is ${formatCurrency(gap)} above next-month trend.`
                : `Your predicted rent is ${formatCurrency(Math.abs(gap))} below next-month trend.`;
        }

        const timelineNote = isSyntheticTimeline ? 'Timeline inferred from available records.' : '';
        summaryEl.textContent = `Trend: ${trendLabel} | Avg monthly change: ${formatCurrency(avgChange)} | Next month forecast: ${formatCurrency(nextMonth)}${comparisonText ? ` | ${comparisonText}` : ''}${timelineNote ? ` | ${timelineNote}` : ''}`;
        box.style.display = 'block';
        renderDecisionOverview();
    } catch (err) {
        latestTrendForecast = null;
        renderDecisionOverview();
        console.error('Error loading rent trend forecast:', err);
    }
}

function initFeaturePages() {
    const bindClick = (id, handler) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('click', handler);
    };

    bindClick('trnd-load-btn', () => loadTrendsPage());
    bindClick('trnd-use-current-btn', () => useCurrentSearchForTrends());
    bindClick('compare-load-btn', () => loadCompareListings());
    bindClick('compare-top-btn', () => fillCompareFromRecommendations());
    bindClick('compare-shortlist-fill-btn', () => fillCompareFromShortlist());
    bindClick('score-load-btn', () => loadLocalityScorecard());
    bindClick('score-use-current-btn', () => useCurrentSearchForScorecard());
    bindClick('planner-refresh-btn', () => refreshPlannerData());
    bindClick('save-search-btn', () => saveCurrentSearch());
    bindClick('analytics-save-search-btn', () => saveCurrentSearch());
    bindClick('analytics-open-trends-btn', () => useCurrentSearchForTrends());
    bindClick('analytics-open-compare-btn', () => openCompareFromAnalytics());
    bindClick('analytics-open-score-btn', () => useCurrentSearchForScorecard());
    bindClick('analytics-open-planner-btn', () => openPlannerFromAnalytics());
    bindClick('feedback-save-btn', () => savePredictionFeedback());

    const recommendationsList = document.getElementById('recommendations-list');
    if (recommendationsList) {
        recommendationsList.addEventListener('click', async (event) => {
            const mapButton = event.target.closest('.focus-map-btn');
            if (mapButton) {
                focusMapLocation(mapButton.getAttribute('data-lat'), mapButton.getAttribute('data-lon'));
                return;
            }

            const compareButton = event.target.closest('.add-compare-btn');
            if (compareButton) {
                addListingIdToCompare(compareButton.getAttribute('data-id'), true);
                return;
            }

            const button = event.target.closest('.save-shortlist-btn');
            if (!button) return;
            const payload = safeText(button.getAttribute('data-shortlist'));
            if (!payload) return;
            try {
                await saveShortlistItem(JSON.parse(decodeURIComponent(payload)));
            } catch (err) {
                console.error('Shortlist action failed:', err);
            }
        });
    }

    const shortlistList = document.getElementById('shortlist-result');
    if (shortlistList) {
        shortlistList.addEventListener('click', async (event) => {
            const compareButton = event.target.closest('.compare-shortlist-btn');
            if (compareButton) {
                addListingIdToCompare(compareButton.getAttribute('data-id'), true);
                return;
            }

            const button = event.target.closest('.remove-shortlist-btn');
            if (!button) return;
            try {
                await deleteShortlistItem(button.getAttribute('data-id'));
            } catch (err) {
                console.error('Delete shortlist failed:', err);
            }
        });
    }

    const savedSearchList = document.getElementById('saved-search-result');
    if (savedSearchList) {
        savedSearchList.addEventListener('click', async (event) => {
            const applyButton = event.target.closest('.apply-search-btn');
            if (applyButton) {
                const payload = safeText(applyButton.getAttribute('data-search'));
                if (payload) applySavedSearch(JSON.parse(decodeURIComponent(payload)));
                return;
            }

            const deleteButton = event.target.closest('.delete-search-btn');
            if (deleteButton) {
                try {
                    await deleteSavedSearch(deleteButton.getAttribute('data-id'));
                } catch (err) {
                    console.error('Delete saved search failed:', err);
                }
            }
        });
    }
}

function safeText(value) {
    return value === null || value === undefined ? '' : String(value);
}

function normalizeOptionalText(value) {
    const text = safeText(value).trim();
    return ['nan', 'none', 'null', 'undefined'].includes(text.toLowerCase()) ? '' : text;
}

function renderCompactMeta(items) {
    const safeItems = (items || []).filter((item) => safeText(item).trim());
    if (safeItems.length === 0) return '';
    return `
        <div class="compact-card-meta">
            ${safeItems.map((item) => `<span>${escapeHtml(item)}</span>`).join('')}
        </div>
    `;
}

function renderListingOperationalBadges(freshnessLabel, freshnessClass, contactLabel, contactClass) {
    const badges = [];
    const freshnessText = safeText(freshnessLabel).trim();
    const contactText = safeText(contactLabel).trim();

    if (freshnessText && !/unavailable/i.test(freshnessText)) {
        badges.push(`<span class="freshness-badge ${escapeHtml(safeText(freshnessClass).trim() || 'freshness-unknown')}">${escapeHtml(freshnessText)}</span>`);
    }

    if (contactText && !/unavailable/i.test(contactText)) {
        badges.push(`<span class="contact-badge ${escapeHtml(safeText(contactClass).trim() || 'contact-unknown')}">${escapeHtml(contactText)}</span>`);
    }

    return badges.join('');
}

function renderSavedSearchCard(item) {
    const predictionInputs = item.search_params?.prediction_inputs || {};
    const predictedRent = item.search_params?.latest_predicted_rent;
    return `
        <div class="mini-card planner-card compact-card">
            <div class="compact-card-eyebrow">Saved search</div>
            <div class="planner-card-head compact-card-head">
                <div>
                    <strong class="compact-card-title">${escapeHtml(item.name)}</strong>
                    <div class="compact-card-subtitle">${escapeHtml(item.created_at || '')}</div>
                </div>
                <div class="planner-value compact-card-value">${predictedRent ? formatCurrency(predictedRent) : 'Pending'}</div>
            </div>
            ${renderCompactMeta([
                predictionInputs.city || 'City n/a',
                `${safeText(predictionInputs.bhk || 'Any')} BHK`,
                predictionInputs.locality || '',
                predictionInputs.budget_target ? `Budget ${formatCurrency(predictionInputs.budget_target)}` : 'No budget'
            ])}
            <div class="recommendation-actions">
                <button class="btn-link apply-search-btn" type="button" data-search="${escapeHtml(encodeURIComponent(JSON.stringify(item.search_params || {})))}">Apply</button>
                <button class="btn-link delete-search-btn" type="button" data-id="${item.id}">Delete</button>
            </div>
        </div>
    `;
}

function renderShortlistCard(item) {
    return `
        <div class="mini-card planner-card compact-card">
            <div class="compact-card-eyebrow">Shortlist</div>
            <div class="planner-card-head compact-card-head">
                <div>
                    <strong class="compact-card-title">${escapeHtml(item.locality || 'Saved listing')}</strong>
                    <div class="compact-card-subtitle">${escapeHtml(item.city || 'Unknown city')}</div>
                </div>
                <div class="planner-value compact-card-value">${item.rent ? formatCurrency(item.rent) : 'Saved'}</div>
            </div>
            ${renderCompactMeta([
                item.bhk ? `${item.bhk} BHK` : 'Any BHK',
                item.size ? `${item.size} sqft` : 'Size n/a',
                item.created_at || ''
            ])}
            ${item.notes ? `<div class="recommendation-note compact-card-note">${escapeHtml(item.notes)}</div>` : ''}
            <div class="recommendation-actions">
                ${item.listing_id ? `<button class="btn-link compare-shortlist-btn" type="button" data-id="${item.listing_id}">Compare</button>` : ''}
                <button class="btn-link remove-shortlist-btn" type="button" data-id="${item.id}">Remove</button>
            </div>
        </div>
    `;
}

function renderCompareCard(row) {
    const locality = safeText(row['Area Locality']) || 'Unknown locality';
    const city = safeText(row.City) || 'Unknown city';
    const bhk = safeText(row.BHK).trim();
    const size = safeText(row.Size).trim();
    const furnishing = safeText(row['Furnishing Status']).trim();
    const areaType = safeText(row['Area Type']).trim();
    const flags = (row.trust_flags || [])
        .slice(0, 3)
        .map((flag) => `<span class="flag-chip">${escapeHtml(safeText(flag).replace(/_/g, ' '))}</span>`)
        .join('');
    const operationalBadges = renderListingOperationalBadges(
        row.freshness_label,
        row.freshness_class,
        row.contact_type_label,
        row.contact_type_class
    );

    return `
        <div class="mini-card compact-card">
            <div class="compact-card-eyebrow">Listing #${escapeHtml(row.id)}</div>
            <div class="compact-card-head">
                <div>
                    <strong class="compact-card-title">${escapeHtml(locality)}</strong>
                    <div class="compact-card-subtitle">${escapeHtml(city)}</div>
                </div>
                <div class="compact-card-value">${formatCurrency(row.Rent)}</div>
            </div>
            ${renderCompactMeta([
                bhk ? `${bhk} BHK` : '',
                size ? `${size} sqft` : '',
                furnishing,
                areaType
            ])}
            <div class="chip-row">
                <span class="trust-badge ${getTrustClass(row.trust_score)}">${escapeHtml(getTrustLabel(row.trust_score))}</span>
                ${operationalBadges}
                ${flags}
            </div>
        </div>
    `;
}

function renderScorecardCard(row) {
    const localityScore = Number(row.locality_score || 0);
    const livability = Number(row.livability || 0);
    return `
        <div class="mini-card compact-card">
            <div class="compact-card-eyebrow">Rank #${escapeHtml(row.rank)}</div>
            <div class="compact-card-head">
                <div>
                    <strong class="compact-card-title">${escapeHtml(safeText(row.locality))}</strong>
                    <div class="compact-card-subtitle">${escapeHtml(safeText(row.city))}</div>
                </div>
                <div class="compact-card-score">${localityScore.toFixed(0)}%</div>
            </div>
            ${renderCompactMeta([
                `Avg ${formatCurrency(row.avg_rent)}`,
                `Livability ${livability.toFixed(1)}/10`,
                `Listings ${safeText(row.listings || 0)}`
            ])}
        </div>
    `;
}

function setInputValue(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value === null || value === undefined ? '' : value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
}

async function refreshPlannerData() {
    await Promise.all([
        refreshShortlist(),
        refreshSavedSearches()
    ]);
}

async function saveShortlistItem(payload) {
    const response = await apiFetch('/api/shortlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || 'Failed to save shortlist item');
    }
    await refreshShortlist();
    return data;
}

async function refreshShortlist() {
    const result = document.getElementById('shortlist-result');
    if (!result) return;
    result.innerHTML = 'Loading shortlist...';
    try {
        const res = await apiFetch('/api/shortlist');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load shortlist');
        const items = data.items || [];
        if (items.length === 0) {
            result.innerHTML = 'No shortlist items saved yet.';
            return;
        }
        result.innerHTML = items.map((item) => renderShortlistCard(item)).join('');
    } catch (err) {
        result.innerHTML = `Error: ${escapeHtml(err.message)}`;
    }
}

async function deleteShortlistItem(id) {
    if (!id) return;
    const res = await apiFetch(`/api/shortlist?id=${encodeURIComponent(id)}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to delete shortlist item');
    await refreshShortlist();
}

async function saveCurrentSearch() {
    const resolvedName = getActiveSavedSearchName();
    const payload = {
        name: resolvedName,
        search_params: buildCurrentSearchPayload()
    };
    setSavedSearchStatus('Saving current search...');
    try {
        const res = await apiFetch('/api/saved_searches', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to save search');
        const savedName = safeText(data.item?.name || resolvedName || 'search');
        setSavedSearchNameValue(savedName);
        setSavedSearchStatus(`Saved ${savedName}.`);
        await refreshSavedSearches();
    } catch (err) {
        setSavedSearchStatus(`Error: ${err.message}`);
    }
}

async function refreshSavedSearches() {
    const result = document.getElementById('saved-search-result');
    if (!result) return;
    result.innerHTML = 'Loading saved searches...';
    try {
        const res = await apiFetch('/api/saved_searches');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load saved searches');
        const items = data.items || [];
        if (items.length === 0) {
            result.innerHTML = 'No saved searches yet.';
            return;
        }
        result.innerHTML = items.map((item) => renderSavedSearchCard(item)).join('');
    } catch (err) {
        result.innerHTML = `Error: ${escapeHtml(err.message)}`;
    }
}

function applySavedSearch(searchParams) {
    const predictionInputs = searchParams?.prediction_inputs || {};
    const plannerPreferences = searchParams?.planner_preferences || {};
    const costAssumptions = searchParams?.cost_assumptions || {};

    setInputValue('city', predictionInputs.city || '');
    setInputValue('budget_target', predictionInputs.budget_target ?? 35000);
    setInputValue('locality', predictionInputs.locality || '');
    setInputValue('bhk', predictionInputs.bhk || 2);
    setInputValue('size', predictionInputs.size || 850);
    setInputValue('bathroom', predictionInputs.bathroom || 2);
    setInputValue('area_type', predictionInputs.area_type || 'Super Area');
    setInputValue('furnishing', predictionInputs.furnishing || 'Semi-Furnished');
    setInputValue('tenant', predictionInputs.tenant || plannerPreferences.tenant || 'Bachelors/Family');
    setInputValue('bathroom_type', predictionInputs.bathroom_type || 'Standard');

    setInputValue('work_city', plannerPreferences.work_city || '');
    const restoredWorkMapUrl = safeText(plannerPreferences.work_map_url).trim()
        || buildMapUrlFromCoordinates(plannerPreferences.work_lat, plannerPreferences.work_lon);
    setInputValue('work_map_url', restoredWorkMapUrl);
    setInputValue('work_lat', plannerPreferences.work_lat ?? '');
    setInputValue('work_lon', plannerPreferences.work_lon ?? '');
    syncWorkLocationFromMapUrl({ showStatus: Boolean(restoredWorkMapUrl) });
    setInputValue('has_pet', plannerPreferences.has_pet ? 'true' : 'false');
    setInputValue('cost_weight', plannerPreferences.cost_weight ?? 35);
    setInputValue('commute_weight', plannerPreferences.commute_weight ?? 15);
    setInputValue('safety_weight', plannerPreferences.safety_weight ?? 20);
    setInputValue('transit_weight', plannerPreferences.transit_weight ?? 15);
    setInputValue('amenity_weight', plannerPreferences.amenity_weight ?? 15);

    setInputValue('deposit_months', costAssumptions.deposit_months ?? 2);
    setInputValue('brokerage_months', costAssumptions.brokerage_months ?? 1);
    setInputValue('maintenance', costAssumptions.maintenance ?? 2500);
    setInputValue('utilities', costAssumptions.utilities ?? 3000);
    setInputValue('parking', costAssumptions.parking ?? 1500);
    setInputValue('moving_cost', costAssumptions.moving_cost ?? 8000);

    syncLinkedSectionsFromCurrentPlan({ forceCompare: true });
    refreshSavedSearchNameSuggestions();
    refreshMapVisualization();
    showSection('predict');
}

async function deleteSavedSearch(id) {
    if (!id) return;
    const res = await apiFetch(`/api/saved_searches?id=${encodeURIComponent(id)}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to delete saved search');
    await refreshSavedSearches();
}

async function savePredictionFeedback() {
    const status = document.getElementById('feedback-status');
    if (status) status.textContent = 'Saving feedback...';
    try {
        const predictedRent = parseNumberOrNull(document.getElementById('feedback-predicted')?.value);
        const actualRent = parseNumberOrNull(document.getElementById('feedback-actual')?.value);
        if (predictedRent === null || actualRent === null) {
            throw new Error('Predicted and actual rent are required.');
        }

        const res = await apiFetch('/api/prediction_feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: latestPredictionInput || collectPredictFormInput(),
                predicted_rent: predictedRent,
                actual_rent: actualRent,
                feedback_text: safeText(document.getElementById('feedback-text')?.value).trim()
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to save feedback');
        if (status) status.textContent = `Feedback saved at ${safeText(data.feedback?.created_at || '')}.`;
        setInputValue('feedback-actual', '');
        setInputValue('feedback-text', '');
    } catch (err) {
        if (status) status.textContent = `Error: ${err.message}`;
    }
}

async function loadTrendsPage() {
    const city = safeText(document.getElementById('trnd-city')?.value).trim();
    const bhk = parseInt(document.getElementById('trnd-bhk')?.value || '2', 10);
    const locality = safeText(document.getElementById('trnd-locality')?.value).trim();
    const horizon = parseInt(document.getElementById('trnd-horizon')?.value || '6', 10);
    const summary = document.getElementById('trnd-summary');
    const canvas = document.getElementById('trndChart');
    if (!summary || !canvas) return;

    summary.textContent = 'Loading trend forecast...';
    try {
        const params = new URLSearchParams({
            city,
            bhk: String(Number.isFinite(bhk) ? bhk : 2),
            locality,
            months_history: '12',
            months_forecast: String(Number.isFinite(horizon) ? horizon : 6)
        });
        const res = await apiFetch(`/api/rent_trends?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to load trend data');
        const data = await res.json();
        const historical = data.historical || [];
        const forecast = data.forecast || [];
        if (historical.length === 0) {
            summary.textContent = 'No trend data available for selected filters.';
            return;
        }

        const labels = historical.map((x) => x.month).concat(forecast.map((x) => x.month));
        const histSeries = historical.map((x) => x.median_rent).concat(new Array(forecast.length).fill(null));
        const forecastSeries = new Array(labels.length).fill(null);
        forecastSeries[historical.length - 1] = historical[historical.length - 1].median_rent;
        forecast.forEach((x, idx) => {
            forecastSeries[historical.length + idx] = x.forecast_rent;
        });

        if (trendsPageChart) trendsPageChart.destroy();
        trendsPageChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Historical Median',
                        data: histSeries,
                        borderColor: '#22c55e',
                        borderWidth: 2,
                        tension: 0.28,
                        pointRadius: 2
                    },
                    {
                        label: 'Forecast',
                        data: forecastSeries,
                        borderColor: '#f59e0b',
                        borderWidth: 2,
                        borderDash: [6, 4],
                        tension: 0.2,
                        pointRadius: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.15)' } },
                    y: {
                        ticks: { color: '#94a3b8', callback: (val) => `Rs ${Math.round(val / 1000)}k` },
                        grid: { color: 'rgba(148,163,184,0.15)' }
                    }
                }
            }
        });
        applyChartTheme(trendsPageChart);

        summary.textContent = `Trend ${data.trend_direction || 'stable'} | Next month forecast ${formatCurrency(data.next_month_forecast || 0)} | Samples ${data.market_size || 0}`;
    } catch (err) {
        summary.textContent = `Error: ${err.message}`;
    }
}

async function loadCompareListings() {
    const ids = safeText(document.getElementById('compare-ids')?.value).trim();
    const summary = document.getElementById('compare-summary');
    const result = document.getElementById('compare-result');
    if (!summary || !result) return;
    if (!ids) {
        ensureCompareHelperText();
        result.innerHTML = '';
        return;
    }
    summary.textContent = 'Loading comparison...';
    result.innerHTML = '';
    try {
        const res = await apiFetch(`/api/compare_listings?ids=${encodeURIComponent(ids)}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Compare failed');
        const rows = data.comparisons || [];
        const s = data.summary || {};
        summary.textContent = `Compared ${s.count || rows.length} listings | Rent range: ${formatCurrency(s.min_rent || 0)} - ${formatCurrency(s.max_rent || 0)}`;
        result.innerHTML = rows.map((row) => renderCompareCard(row)).join('');
    } catch (err) {
        summary.textContent = `Error: ${err.message}`;
    }
}

async function loadLocalityScorecard() {
    const city = safeText(document.getElementById('score-city')?.value).trim();
    const bhk = parseInt(document.getElementById('score-bhk')?.value || '2', 10);
    const result = document.getElementById('score-result');
    if (!result) return;
    result.innerHTML = 'Loading scorecard...';
    try {
        const params = new URLSearchParams({ city, bhk: String(Number.isFinite(bhk) ? bhk : 2), limit: '15' });
        const res = await apiFetch(`/api/locality_scorecard?${params.toString()}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Scorecard failed');
        const rows = data.scorecard || [];
        if (rows.length === 0) {
            result.innerHTML = 'No localities found for the selected filters.';
            return;
        }
        result.innerHTML = rows.map((row) => renderScorecardCard(row)).join('');
    } catch (err) {
        result.innerHTML = `Error: ${err.message}`;
    }
}

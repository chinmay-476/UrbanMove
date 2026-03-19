document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initAnalyticsModeControls();
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
const WORK_MAP_STATUS_DEFAULT = 'Paste a pinned map link and the app will read latitude and longitude automatically.';
const FALLBACK_API_BASES = ['http://127.0.0.1:5000', 'http://localhost:5000'];
let currentTheme = 'dark';
let resolvedApiBase = null;

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

function applyAnalyticsMode(mode, options = {}) {
    const predictSection = document.getElementById('predict');
    if (!predictSection) return;

    const normalizedMode = mode === 'glass' ? 'glass' : 'classic';
    predictSection.dataset.analyticsMode = normalizedMode;
    updateAnalyticsModeButtons(normalizedMode);

    if (options.persist !== false) {
        writeStoredAnalyticsMode(normalizedMode);
    }
}

function initAnalyticsModeControls() {
    applyAnalyticsMode(readStoredAnalyticsMode(), { persist: false });

    document.querySelectorAll('button[data-analytics-mode]').forEach((button) => {
        button.addEventListener('click', () => applyAnalyticsMode(button.dataset.analyticsMode));
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
    if (typeof event !== 'undefined' && event && event.currentTarget) {
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
    // Center on India broadly
    map = L.map('map').setView([22.3511, 78.6677], 5);

    // Click anywhere on the map to open that point directly in Google Maps
    map.on('click', (e) => {
        const { lat, lng } = e.latlng;
        const gmapsUrl = `https://www.google.com/maps?q=${lat},${lng}`;
        window.open(gmapsUrl, '_blank', 'noopener');
    });

    // Create dedicated pane for heatmap under markers
    map.createPane('heatmapPane');
    map.getPane('heatmapPane').style.zIndex = 300;
    baseMapTileLayer = L.tileLayer(getThemeTileUrl(), {
        attribution: '(c) CARTO'
    });
    baseMapTileLayer.addTo(map);

    try {
        const res = await apiFetch('/api/map_data');
        const data = await res.json();

        // Heatmap Data: [lat, lng, intensity]
        // Intensity based on Neighborhood_Livability_Score (normalized if needed, but heat layer handles it usually)
        const heatPoints = data.map(p => [p.Latitude, p.Longitude, p.Neighborhood_Livability_Score ? p.Neighborhood_Livability_Score * 10 : 0.5]);

        if (L.heatLayer) {
            L.heatLayer(heatPoints, {
                radius: 25,
                blur: 15,
                maxZoom: 17,
                gradient: { 0.2: '#0ea5e9', 0.4: '#22c55e', 0.7: '#f59e0b', 1: '#ef4444' },
                pane: 'heatmapPane'
            }).addTo(map);
        }

        // Add Circle Markers with labels for detailed info interaction
        const markers = [];
        data.forEach(p => {
            // Color marker based on Rent (red for expensive, blue for affordable)
            const rent = p.Rent || 0;
            const color = rent > 50000 ? '#ef4444' : rent > 30000 ? '#f59e0b' : '#6366f1';
            const locationName = p['Area Locality'] || p.City || 'Unknown Location';
            const gmapsUrl = `https://www.google.com/maps?q=${p.Latitude},${p.Longitude}`;
            
            // Create marker with label
            const marker = L.circleMarker([p.Latitude, p.Longitude], {
                radius: 6,
                fillColor: color,
                color: '#fff',
                weight: 1.5,
                fillOpacity: 0.7,
                className: 'marker-point',
                pane: 'markerPane', // ensure marker sits above heat layer
                zIndexOffset: 500
            });
            
            // Build comprehensive popup content
            let popupContent = `
                <div style="color: #1e293b; font-family: 'Inter', sans-serif; min-width: 200px;">
                    <h4 style="margin: 0 0 8px 0; padding-bottom: 6px; border-bottom: 2px solid #6366f1; color: #0f172a; font-size: 1.1em;">
                         ${locationName}
                    </h4>
                    <div style="margin-bottom: 6px;"><strong>City:</strong> ${p.City || 'N/A'}</div>
                    <div style="margin-bottom: 6px; font-size: 1.2em; color: #059669; font-weight: 600;">
                         <strong>Rent:</strong> ${formatCurrency(rent)}
                    </div>
            `;
            
            if (p.BHK) {
                popupContent += `<div style="margin-bottom: 4px;"><strong>BHK:</strong> ${p.BHK}</div>`;
            }
            if (p.Size) {
                popupContent += `<div style="margin-bottom: 4px;"><strong>Size:</strong> ${p.Size} sqft</div>`;
            }
            if (p['Furnishing Status']) {
                popupContent += `<div style="margin-bottom: 4px;"><strong>Furnishing:</strong> ${p['Furnishing Status']}</div>`;
            }
            if (p['Area Type']) {
                popupContent += `<div style="margin-bottom: 4px;"><strong>Area Type:</strong> ${p['Area Type']}</div>`;
            }
            if (p.Neighborhood_Livability_Score) {
                const score = parseFloat(p.Neighborhood_Livability_Score).toFixed(1);
                const scoreColor = score >= 7 ? '#059669' : score >= 5 ? '#f59e0b' : '#ef4444';
                popupContent += `<div style="margin-bottom: 4px;">
                    <strong>Livability Score:</strong> 
                    <span style="color: ${scoreColor}; font-weight: 600;">${score}/10</span>
                </div>`;
            }
            if (p.Cluster_ID !== undefined) {
                popupContent += `<div style="margin-bottom: 4px;"><strong>Cluster Zone:</strong> ${p.Cluster_ID}</div>`;
            }
            
            popupContent += `
                    <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #e2e8f0; font-size: 0.75em; color: #64748b;">
                         Coordinates: ${p.Latitude.toFixed(4)}, ${p.Longitude.toFixed(4)}
                        <div style="margin-top: 6px;">
                            <a href="${gmapsUrl}" target="_blank" rel="noopener noreferrer" style="color: #2563eb; font-weight: 600; text-decoration: none;">Open in Google Maps -></a>
                        </div>
                    </div>
                </div>
            `;
            
            marker.bindPopup(popupContent);
            
            // Add label/tooltip showing location name
            marker.bindTooltip(
                `<div style="font-weight: 600; font-size: 0.85em;">${locationName}<br/>${formatCurrency(rent)}</div>`,
                {
                    permanent: false,
                    direction: 'top',
                    offset: [0, -10],
                    className: 'map-label'
                }
            );
            
            // Open Google Maps directly on marker click
            marker.on('click', (ev) => {
                // Stop Leaflet default behavior interfering with navigation
                if (ev.originalEvent) {
                    ev.originalEvent.preventDefault();
                    ev.originalEvent.stopPropagation();
                }
                window.open(gmapsUrl, '_blank', 'noopener');
            });
            
            marker.addTo(map);
            if (marker.bringToFront) marker.bringToFront(); // ensure markers are above heat layer
            markers.push(marker);
        });
        
        // Add legend for the heatmap
        const legend = L.control({position: 'bottomright'});
        legend.onAdd = function(map) {
            const div = L.DomUtil.create('div', 'map-legend');
            div.style.backgroundColor = 'rgba(30, 41, 59, 0.9)';
            div.style.padding = '12px';
            div.style.borderRadius = '8px';
            div.style.color = '#f8fafc';
            div.style.fontSize = '0.85em';
            div.style.fontFamily = 'Inter, sans-serif';
            div.innerHTML = `
                <h4 style="margin: 0 0 8px 0; font-size: 0.9em; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 4px;">Heatmap (Livability)</h4>
                <div style="margin-bottom: 4px;">* Low</div>
                <div style="margin-bottom: 4px;">* Medium</div>
                <div style="margin-bottom: 4px;">* High</div>
                <div style="margin-bottom: 8px;">* Very High</div>
                <h4 style="margin: 8px 0 4px 0; font-size: 0.9em; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;">Marker Colors</h4>
                <div style="margin-bottom: 4px;">* Affordable (&lt;Rs 30k)</div>
                <div style="margin-bottom: 4px;">* Moderate (Rs 30k-Rs 50k)</div>
                <div>* Expensive (&gt;Rs 50k)</div>
            `;
            return div;
        };
        legend.addTo(map);

    } catch (err) {
        console.error("Map data error", err);
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
        input.addEventListener('input', () => scheduleRecommendationRefresh());
        input.addEventListener('change', () => scheduleRecommendationRefresh());
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
            const mapsUrl = `https://www.google.com/maps?q=${item.coordinates.lat},${item.coordinates.lon}`;
            const profileBadge = item.profile_source === 'curated' ? 'Curated profile' : 'Fallback profile';
            const profileNotes = normalizeOptionalText(item.profile_notes);
            const listingId = parseInt(item.sample_listing_id, 10);
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
                        <a class="btn-link" href="${mapsUrl}" target="_blank" rel="noopener noreferrer">Open map</a>
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

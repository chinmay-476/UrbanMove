document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    initMap();
    initForm();
    loadCities();
    initTrendForecastControls();
});

let latestPredictionInput = null;
let latestPredictedRent = null;
let trendForecastChart = null;

// Load cities and localities dynamically
async function loadCities() {
    try {
        const res = await fetch('/api/cities');
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
        }
        
        // Update locality input with autocomplete suggestions (optional enhancement)
        // For now, we'll just store the data for potential future use
        window.citiesData = data;
    } catch (err) {
        console.error("Error loading cities:", err);
    }
}

// Navigation
function showSection(id) {
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    document.getElementById(id).classList.add('active');
    event.currentTarget.classList.add('active');

    if (id === 'map-view') {
        setTimeout(() => map.invalidateSize(), 100);
    }
}

// Stats
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
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
        }
    } catch (err) {
        console.error("Stats error", err);
        document.getElementById('avg-rent-stat').textContent = "Error";
    }
}

// Map
let map;
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
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '(c) CARTO'
    }).addTo(map);

    try {
        const res = await fetch('/api/map_data');
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

function initTrendForecastControls() {
    const horizonSelect = document.getElementById('trend-horizon');
    if (!horizonSelect) return;

    horizonSelect.addEventListener('change', async () => {
        if (!latestPredictionInput || !latestPredictedRent) return;
        await loadRentTrendForecast(latestPredictionInput, latestPredictedRent);
    });
}

function initForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);

        // Convert FormData to object and ensure numeric fields are numbers
        const data = {
            city: formData.get('city'),
            locality: formData.get('locality') || '',
            bhk: parseInt(formData.get('bhk'), 10),
            size: parseFloat(formData.get('size')),
            bathroom: parseInt(formData.get('bathroom'), 10),
            area_type: formData.get('area_type'),
            furnishing: formData.get('furnishing'),
            tenant: formData.get('tenant'),
            bathroom_type: formData.get('bathroom_type')
        };

        const btn = form.querySelector('button');
        const originalText = btn.textContent;
        btn.textContent = 'Calculating...';
        btn.disabled = true;

        try {
            const res = await fetch('/api/predict', {
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

            resultBox.style.display = 'flex';
            predictionEl.textContent = formatCurrency(result.predicted_rent);

            if (result.location) {
                const disclaimer = resultBox.querySelector('.disclaimer');
                if (disclaimer) {
                    disclaimer.textContent = `Based on historical market data for ${result.location.city}${result.location.locality ? ` - ${result.location.locality}` : ''}`;
                }
            }

            if (result.predicted_rent) {
                latestPredictionInput = data;
                latestPredictedRent = result.predicted_rent;

                await Promise.all([
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

// Load recommended locations from backend based on prediction
async function loadRecommendations(formInput, predictedRent) {
    const box = document.getElementById('recommendations-box');
    const list = document.getElementById('recommendations-list');
    if (!box || !list) return;

    list.innerHTML = '';
    box.style.display = 'none';

    try {
        const city = formInput.city;
        const bhk = formInput.bhk;
        const minRent = Math.max(0, predictedRent * 0.8);
        const maxRent = predictedRent * 1.2;

        const params = new URLSearchParams({
            city: city || '',
            bhk: bhk ? String(bhk) : '',
            min_rent: String(Math.round(minRent)),
            max_rent: String(Math.round(maxRent)),
            limit: '5'
        });

        const res = await fetch(`/api/properties?${params.toString()}`);
        if (!res.ok) {
            console.error('Recommendations fetch failed');
            return;
        }

        const json = await res.json();
        const properties = json.properties || json;

        if (!properties || properties.length === 0) {
            return;
        }

        properties.forEach((p) => {
            const locality = p['Area Locality'] || 'Preferred locality';
            const cityName = p.City || city || 'City';
            const bhkText = p.BHK ? `${p.BHK} BHK` : '';
            const sizeText = p.Size ? `${p.Size} sqft` : '';
            const furnishText = p['Furnishing Status'] || '';
            const areaTypeText = p['Area Type'] || '';
            const rentVal = p.Rent || 0;

            const lat = p.Latitude;
            const lon = p.Longitude;
            const gmapsUrl = (lat && lon)
                ? `https://www.google.com/maps?q=${lat},${lon}`
                : `https://www.google.com/maps?q=${encodeURIComponent(`${locality}, ${cityName}`)}`;

            const card = document.createElement('div');
            card.className = 'recommendation-card';

            const header = document.createElement('div');
            header.className = 'recommendation-card-header';
            header.innerHTML = `
                <div>
                    <div class="recommendation-title">${locality}</div>
                    <div class="recommendation-city">${cityName}</div>
                </div>
                <div class="recommendation-rent">${formatCurrency(rentVal)}</div>
            `;

            const meta = document.createElement('div');
            meta.className = 'recommendation-meta';
            [bhkText, sizeText, furnishText, areaTypeText]
                .filter(Boolean)
                .forEach((text) => {
                    const span = document.createElement('span');
                    span.textContent = text;
                    meta.appendChild(span);
                });

            const actions = document.createElement('div');
            actions.className = 'recommendation-actions';
            const link = document.createElement('a');
            link.className = 'btn-link';
            link.href = gmapsUrl;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = 'View on Google Maps';
            actions.appendChild(link);

            card.appendChild(header);
            if (meta.childNodes.length > 0) {
                card.appendChild(meta);
            }
            card.appendChild(actions);

            list.appendChild(card);
        });

        box.style.display = 'block';
    } catch (err) {
        console.error('Error loading recommendations:', err);
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
        const params = new URLSearchParams({
            city: formInput.city || '',
            bhk: formInput.bhk ? String(formInput.bhk) : '',
            budget: String(Math.round(predictedRent)),
            limit: '5'
        });

        const res = await fetch(`/api/budget_advisor?${params.toString()}`);
        if (!res.ok) {
            console.error('Budget advisor fetch failed');
            return;
        }

        const data = await res.json();
        const recommendations = data.recommendations || [];
        if (recommendations.length === 0) {
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

        const res = await fetch(`/api/market_insights?${params.toString()}`);
        if (!res.ok) {
            console.error('Market insights fetch failed');
            return;
        }

        const data = await res.json();
        if (!data || !data.market_size) {
            return;
        }

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
    } catch (err) {
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

        const res = await fetch(`/api/rent_trends?${params.toString()}`);
        if (!res.ok) {
            console.error('Rent trend fetch failed');
            return;
        }

        const data = await res.json();
        const historical = data.historical || [];
        const forecast = data.forecast || [];
        if (historical.length < 1) {
            return;
        }

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
    } catch (err) {
        console.error('Error loading rent trend forecast:', err);
    }
}

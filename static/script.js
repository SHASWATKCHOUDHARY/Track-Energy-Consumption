/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * EnergyAI — Premium Smart Energy Dashboard JavaScript
 * ═══════════════════════════════════════════════════════════════════════════════
 * Features:
 * - Real-time chart updates (simulated live data)
 * - Smooth page transitions with animations
 * - Theme toggle (dark/light) with localStorage persistence
 * - Toast notifications & alert systems
 * - All API integrations for ML/AI backend
 * ═══════════════════════════════════════════════════════════════════════════════
 */

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════════════════════
let currentTheme = localStorage.getItem('theme') || 'dark';
let chartUpdateInterval = null;
let summaryData = null;

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 EnergyAI Dashboard Initialized');
    
    // Apply saved theme
    applyTheme(currentTheme);
    
    // Setup navigation
    setupNavigation();
    
    // Setup sidebar toggle
    setupSidebar();
    
    // Setup theme toggle
    setupThemeToggle();
    
    // Setup file upload
    setupFileUpload();
    
    // Setup notification panel
    setupNotifications();
    
    // Load initial data
    loadAllData();
    
    // Start real-time updates (simulated)
    startRealTimeUpdates();
    
    // Add page load animation
    document.body.classList.add('loaded');
});

// ═══════════════════════════════════════════════════════════════════════════════
// THEME MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════
function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    currentTheme = theme;
    localStorage.setItem('theme', theme);
    
    const icon = document.getElementById('themeIcon');
    if (icon) {
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
    
    // Update charts with new theme colors
    updateChartsTheme();
}

function setupThemeToggle() {
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
            showToast(`${newTheme === 'dark' ? '🌙' : '☀️'} ${newTheme.charAt(0).toUpperCase() + newTheme.slice(1)} mode activated`);
        });
    }
}

function updateChartsTheme() {
    const isDark = currentTheme === 'dark';
    const textColor = isDark ? '#e0e0e0' : '#333333';
    const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const bgColor = 'transparent';
    
    // Update all existing Plotly charts
    const chartIds = ['dailyChart', 'hourlyChart', 'weeklyChart', 'monthlyChart', 
                      'rollingChart', 'monthTrendChart', 'forecastChart', 'predictionChart',
                      'appliancePie', 'applianceBar', 'anomalyChart', 'featureChart'];
    
    chartIds.forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) {
            Plotly.relayout(id, {
                paper_bgcolor: bgColor,
                plot_bgcolor: bgColor,
                font: { color: textColor },
                xaxis: { gridcolor: gridColor, color: textColor },
                yaxis: { gridcolor: gridColor, color: textColor }
            });
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// NAVIGATION & SIDEBAR
// ═══════════════════════════════════════════════════════════════════════════════
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.getAttribute('data-section');
            navigateToSection(section);
            
            // Mobile: close sidebar
            if (window.innerWidth <= 1024) {
                closeSidebar();
            }
        });
    });
}

function navigateToSection(sectionId) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-section') === sectionId);
    });
    
    // Hide all sections with fade out
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show target section with animation
    const targetSection = document.getElementById(`section-${sectionId}`);
    if (targetSection) {
        setTimeout(() => {
            targetSection.classList.add('active');
            targetSection.scrollTop = 0;
        }, 50);
    }
    
    // Update page title
    const titles = {
        'dashboard': 'Smart Energy Dashboard',
        'analytics': 'Energy Analytics',
        'predictions': 'AI Predictions',
        'appliances': 'Appliance Analysis',
        'anomalies': 'Anomaly Detection',
        'insights': 'AI Insights',
        'recommendations': 'Energy Recommendations',
        'reports': 'Reports',
        'model': 'Model Evaluation'
    };
    
    const pageTitle = document.querySelector('.page-title');
    if (pageTitle && titles[sectionId]) {
        pageTitle.textContent = titles[sectionId];
    }
}

function setupSidebar() {
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
        });
    }
    
    if (overlay) {
        overlay.addEventListener('click', closeSidebar);
    }
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOADING OVERLAY
// ═══════════════════════════════════════════════════════════════════════════════
function showLoading(title = 'Processing', text = 'Please wait...') {
    const overlay = document.getElementById('loadingOverlay');
    const titleEl = document.getElementById('loadingTitle');
    const textEl = document.getElementById('loadingText');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (titleEl) titleEl.textContent = title;
    if (textEl) textEl.textContent = text;
    if (progressBar) progressBar.style.width = '0%';
    if (progressText) progressText.textContent = '0%';
    
    overlay.classList.add('active');
    
    // Animate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            clearInterval(interval);
            progress = 90;
        }
        if (progressBar) progressBar.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${Math.round(progress)}%`;
    }, 200);
    
    return interval;
}

function hideLoading(interval) {
    if (interval) clearInterval(interval);
    
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (progressBar) progressBar.style.width = '100%';
    if (progressText) progressText.textContent = '100%';
    
    setTimeout(() => {
        document.getElementById('loadingOverlay').classList.remove('active');
    }, 300);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOAST & ALERT NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════════════════
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const icon = toast.querySelector('.toast-icon');
    const msgEl = document.getElementById('toastMessage');
    
    // Set type
    toast.className = `toast ${type}`;
    
    // Set icon
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    icon.className = `toast-icon fas ${icons[type] || icons.success}`;
    
    // Set message
    msgEl.textContent = message;
    
    // Show toast
    toast.classList.add('show');
    
    // Hide after 4s
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

function showAlert(message) {
    const alert = document.getElementById('alertNotification');
    const msgEl = document.getElementById('alertMessage');
    
    msgEl.textContent = message;
    alert.classList.add('show');
    
    // Update notification badge
    const badge = document.getElementById('notifBadge');
    if (badge) {
        const count = parseInt(badge.textContent) || 0;
        badge.textContent = count + 1;
        badge.style.display = 'flex';
    }
}

function closeAlert() {
    document.getElementById('alertNotification').classList.remove('show');
}

// ═══════════════════════════════════════════════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════════════════════════════════════════════
function setupFileUpload() {
    const input = document.getElementById('uploadInput');
    if (input) {
        input.addEventListener('change', handleFileUpload);
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const loadingInterval = showLoading('Uploading Data', `Processing ${file.name}...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideLoading(loadingInterval);
        
        if (data.status === 'success') {
            showToast('✅ ' + (data.message || 'Data uploaded successfully!') + ' Training model now...', 'success');
            // Auto-trigger training immediately after upload
            await trainModel();
            loadAllData();
        } else {
            showToast(`❌ ${data.message || 'Upload failed'}`, 'error');
        }
    } catch (err) {
        hideLoading(loadingInterval);
        showToast('❌ Upload error: ' + err.message, 'error');
    }
    
    e.target.value = '';
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════════════════════
async function loadAllData() {
    console.log('📊 Loading all dashboard data...');
    
    await Promise.all([
        loadSummary(),
        loadDailyChart(),
        loadHourlyChart(),
        loadWeeklyChart(),
        loadMonthlyChart(),
        loadTrends(),
        loadAppliances(),
        loadInsights(),
        loadTips()
    ]);
    
    console.log('✅ All data loaded');
}

async function loadSummary() {
    try {
        const response = await fetch('/api/summary');
        const data = await response.json();
        summaryData = data;
        
        // Update stat cards
        updateElement('totalKwh', formatNumber(data.total_kwh, 2));
        updateElement('avgDaily', formatNumber(data.avg_daily, 2));
        updateElement('peakHour', `${data.peak_hour}:00`);
        updateElement('totalDays', data.total_days);
        updateElement('peakDay', data.peak_day || '--');
        updateElement('peakMonth', data.peak_month || '--');
        updateElement('stdKwh', formatNumber(data.std_kwh, 2));
        
        // Update hero stats
        updateElement('heroTotalKwh', formatNumber(data.total_kwh, 0));
        updateElement('heroAvgDaily', formatNumber(data.avg_daily, 1));
        
        // Calculate and show efficiency
        const efficiency = Math.min(95, Math.max(70, 100 - (data.std_kwh / data.avg_daily * 10)));
        updateElement('heroEfficiency', `${Math.round(efficiency)}%`);
        
    } catch (err) {
        console.error('Error loading summary:', err);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHART FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════
const chartConfig = {
    responsive: true,
    displayModeBar: false
};

function getChartLayout(title = '', showLegend = false) {
    const isDark = currentTheme === 'dark';
    return {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 50, r: 30, t: title ? 40 : 20, b: 50 },
        font: { 
            family: 'Poppins, sans-serif',
            color: isDark ? '#e0e0e0' : '#333333',
            size: 12
        },
        xaxis: {
            gridcolor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
            color: isDark ? '#b0b0b0' : '#666666',
            tickfont: { size: 11 }
        },
        yaxis: {
            gridcolor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
            color: isDark ? '#b0b0b0' : '#666666',
            tickfont: { size: 11 }
        },
        showlegend: showLegend,
        legend: {
            bgcolor: 'transparent',
            font: { color: isDark ? '#e0e0e0' : '#333333' }
        }
    };
}

async function loadDailyChart() {
    try {
        const response = await fetch('/api/daily');
        const data = await response.json();
        
        const trace = {
            x: data.dates,
            y: data.values,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: {
                color: '#00d9ff',
                width: 2,
                shape: 'spline'
            },
            fillcolor: 'rgba(0, 217, 255, 0.15)'
        };
        
        Plotly.newPlot('dailyChart', [trace], getChartLayout(), chartConfig);
    } catch (err) {
        console.error('Error loading daily chart:', err);
    }
}

async function loadHourlyChart() {
    try {
        const response = await fetch('/api/hourly');
        const data = await response.json();
        
        const trace = {
            x: data.hours.map(h => `${h}:00`),
            y: data.values,
            type: 'bar',
            marker: {
                color: data.values.map((v, i) => {
                    const maxVal = Math.max(...data.values);
                    const ratio = v / maxVal;
                    return `rgba(138, 43, 226, ${0.4 + ratio * 0.6})`;
                }),
                line: { color: '#8a2be2', width: 1 }
            }
        };
        
        Plotly.newPlot('hourlyChart', [trace], getChartLayout(), chartConfig);
    } catch (err) {
        console.error('Error loading hourly chart:', err);
    }
}

async function loadWeeklyChart() {
    try {
        const response = await fetch('/api/weekly');
        const data = await response.json();
        
        const trace = {
            x: data.days,
            y: data.values,
            type: 'bar',
            marker: {
                color: ['#ff6b6b', '#ffa94d', '#ffd43b', '#69db7c', '#4dabf7', '#748ffc', '#da77f2'],
                line: { color: 'rgba(255,255,255,0.3)', width: 1 }
            }
        };
        
        Plotly.newPlot('weeklyChart', [trace], getChartLayout(), chartConfig);
    } catch (err) {
        console.error('Error loading weekly chart:', err);
    }
}

async function loadMonthlyChart() {
    try {
        const response = await fetch('/api/monthly');
        const data = await response.json();
        
        const trace = {
            x: data.months,
            y: data.values,
            type: 'bar',
            marker: {
                color: '#00d9ff',
                line: { color: '#00a3cc', width: 1 }
            }
        };
        
        Plotly.newPlot('monthlyChart', [trace], getChartLayout(), chartConfig);
    } catch (err) {
        console.error('Error loading monthly chart:', err);
    }
}

async function loadTrends() {
    try {
        const response = await fetch('/api/trends');
        const data = await response.json();
        
        // Rolling average chart - API returns array of {date, avg_kwh}
        if (data.rolling_7d && data.rolling_7d.length > 0) {
            const trace = {
                x: data.rolling_7d.map(r => r.date),
                y: data.rolling_7d.map(r => r.avg_kwh),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#00ffa3', width: 2, shape: 'spline' },
                fill: 'tozeroy',
                fillcolor: 'rgba(0, 255, 163, 0.1)'
            };
            Plotly.newPlot('rollingChart', [trace], getChartLayout(), chartConfig);
        }
        
        // Month trend chart - API returns monthly_trend as [{month, pct_change}]
        if (data.monthly_trend && data.monthly_trend.length > 0) {
            const months = data.monthly_trend.map(m => m.month);
            const pctValues = data.monthly_trend.map(m => m.pct_change);
            const colors = pctValues.map(v => v >= 0 ? '#ff6b6b' : '#00ffa3');
            const trace = {
                x: months,
                y: pctValues,
                type: 'bar',
                marker: { color: colors }
            };
            Plotly.newPlot('monthTrendChart', [trace], getChartLayout(), chartConfig);
        }
        
        // Update trend text from overall_direction
        if (data.overall_direction) {
            const dir = data.overall_direction;
            const slope = data.slope_per_day || 0;
            const avgD = data.avg_daily_kwh || 0;
            const trendText = `Energy trend is ${dir.toUpperCase()} (slope: ${slope} kWh/day, avg daily: ${avgD} kWh).`;
            updateElement('trendText', trendText);
            const banner = document.getElementById('trendBanner');
            if (banner) {
                banner.className = `trend-banner glass-panel ${dir}`;
            }
        }
        
    } catch (err) {
        console.error('Error loading trends:', err);
    }
}

async function loadAppliances() {
    try {
        const response = await fetch('/api/appliances');
        const data = await response.json();
        
        // Pie chart
        const pieTrace = {
            values: data.values,
            labels: data.labels,
            type: 'pie',
            hole: 0.5,
            marker: {
                colors: ['#00d9ff', '#8a2be2', '#00ffa3', '#ff6b6b', '#ffd43b', '#ff8c42']
            },
            textinfo: 'percent',
            textfont: { color: '#ffffff', size: 12 }
        };
        
        const pieLayout = getChartLayout('', true);
        pieLayout.showlegend = true;
        Plotly.newPlot('appliancePie', [pieTrace], pieLayout, chartConfig);
        
        // Bar chart
        if (data.monthly) {
            const traces = data.monthly.appliances.map((appliance, i) => ({
                x: data.monthly.months,
                y: data.monthly.values[i],
                name: appliance,
                type: 'bar',
                marker: { color: ['#00d9ff', '#8a2be2', '#00ffa3', '#ff6b6b', '#ffd43b'][i % 5] }
            }));
            
            const barLayout = getChartLayout('', true);
            barLayout.barmode = 'stack';
            Plotly.newPlot('applianceBar', traces, barLayout, chartConfig);
        }
        
    } catch (err) {
        console.error('Error loading appliances:', err);
    }
}

async function loadInsights() {
    try {
        const response = await fetch('/api/insights');
        const data = await response.json();
        
        const grid = document.getElementById('insightsGrid');
        if (!grid || data.status === 'error') return;
        
        const insights = data.insights || [];
        
        if (insights.length === 0) {
            grid.innerHTML = '<div class="insight-card glass-panel"><p>No insights available yet.</p></div>';
            return;
        }
        
        grid.innerHTML = insights.map(insight => `
            <div class="insight-card glass-panel ${insight.severity || 'info'}">
                <div class="insight-icon ${insight.severity || 'info'}">
                    <span class="insight-emoji">${insight.icon || '💡'}</span>
                </div>
                <div class="insight-content">
                    <h4>${insight.title || 'Insight'}</h4>
                    <p>${insight.text}</p>
                </div>
            </div>
        `).join('');
        
    } catch (err) {
        console.error('Error loading insights:', err);
    }
}

async function loadTips() {
    try {
        const response = await fetch('/api/tips');
        const tips = await response.json();
        
        const grid = document.getElementById('tipsGrid');
        if (!grid || !Array.isArray(tips)) return;
        
        grid.innerHTML = tips.map((tip, i) => `
            <div class="tip-card glass-panel priority-${tip.priority || 'medium'}">
                <div class="tip-icon">
                    <span class="insight-emoji">${tip.icon || '💡'}</span>
                </div>
                <div class="tip-content">
                    <h4>${tip.title || 'Tip'}</h4>
                    <p>${tip.detail || ''}</p>
                    <div class="tip-saving"><i class="fas fa-leaf"></i> Save ~${tip.saving_pct || 0}%</div>
                </div>
            </div>
        `).join('');
        
    } catch (err) {
        console.error('Error loading tips:', err);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AI MODEL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════
async function trainModel() {
    const loadingInterval = showLoading('Training AI Model', 'Analyzing energy patterns with Random Forest...');
    
    try {
        const response = await fetch('/api/train', { method: 'POST' });
        const data = await response.json();
        
        hideLoading(loadingInterval);
        
        if (data.status === 'success') {
            showToast('🎉 AI Model trained successfully!', 'success');
            displayMetrics(data.metrics);
            
            // Load feature importance separately
            loadFeatureImportance();
            
            // Update predicted usage
            if (data.metrics && data.metrics.predicted_7day) {
                updateElement('predictedUsage', formatNumber(data.metrics.predicted_7day, 1) + ' kWh');
            }
        } else {
            showToast(`❌ Training failed: ${data.message || 'Unknown error'}`, 'error');
        }
        
    } catch (err) {
        hideLoading(loadingInterval);
        showToast('❌ Training error: ' + err.message, 'error');
    }
}

function displayMetrics(metrics) {
    const panel = document.getElementById('metricsPanel');
    if (!panel || !metrics) return;
    
    // Get the best model's metrics (random_forest or linear_regression)
    const selected = metrics.selected || 'random_forest';
    const modelKey = selected.toLowerCase().replace(' ', '_');
    const m = metrics[modelKey] || metrics.random_forest || metrics.linear_regression || {};
    
    // Handle both uppercase (API) and lowercase key formats
    const r2 = m.R2 ?? m.r2 ?? 0;
    const rmse = m.RMSE ?? m.rmse ?? 0;
    const mae = m.MAE ?? m.mae ?? 0;
    const mape = m.MAPE ?? m.mape ?? 0;
    
    panel.innerHTML = `
        <div class="metrics-grid">
            <div class="metric-card glass-panel">
                <div class="metric-icon gradient-green"><i class="fas fa-bullseye"></i></div>
                <div class="metric-info">
                    <span class="metric-value">${(r2 * 100).toFixed(1)}%</span>
                    <span class="metric-label">R² Score</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: ${r2 * 100}%"></div>
                </div>
            </div>
            <div class="metric-card glass-panel">
                <div class="metric-icon gradient-orange"><i class="fas fa-ruler"></i></div>
                <div class="metric-info">
                    <span class="metric-value">${rmse.toFixed(4)}</span>
                    <span class="metric-label">RMSE</span>
                </div>
            </div>
            <div class="metric-card glass-panel">
                <div class="metric-icon gradient-purple"><i class="fas fa-chart-bar"></i></div>
                <div class="metric-info">
                    <span class="metric-value">${mae.toFixed(4)}</span>
                    <span class="metric-label">MAE</span>
                </div>
            </div>
            <div class="metric-card glass-panel">
                <div class="metric-icon gradient-cyan"><i class="fas fa-percentage"></i></div>
                <div class="metric-info">
                    <span class="metric-value">${mape.toFixed(1)}%</span>
                    <span class="metric-label">MAPE</span>
                </div>
            </div>
        </div>
        <div class="model-info glass-panel" style="margin-top: 20px; padding: 16px; text-align: center;">
            <span style="color: var(--text-secondary);">Selected Model:</span>
            <strong style="color: var(--cyan); margin-left: 8px;">${selected}</strong>
            <span style="color: var(--text-muted); margin-left: 16px;">
                Train: ${metrics.train_size || '--'} | Test: ${metrics.test_size || '--'}
            </span>
        </div>
    `;
}

function displayFeatureImportance(features) {
    if (!features || features.length === 0) return;
    
    const sorted = [...features].sort((a, b) => b.importance - a.importance);
    
    const trace = {
        y: sorted.map(f => f.name),
        x: sorted.map(f => f.importance),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map((_, i) => `hsl(${180 + i * 20}, 70%, 50%)`),
            line: { color: 'rgba(255,255,255,0.3)', width: 1 }
        }
    };
    
    const layout = getChartLayout();
    layout.margin.l = 120;
    
    Plotly.newPlot('featureChart', [trace], layout, chartConfig);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORECAST & PREDICTION
// ═══════════════════════════════════════════════════════════════════════════════
async function runForecast() {
    const btn = document.getElementById('forecastBtn');
    if (btn) btn.disabled = true;
    
    const loadingInterval = showLoading('Running Forecast', 'Generating 14-day energy forecast...');
    
    try {
        const response = await fetch('/api/forecast', { method: 'POST' });
        const data = await response.json();
        
        hideLoading(loadingInterval);
        
        if (data.status === 'error') {
            showToast(`❌ ${data.message || 'Forecast failed'}`, 'error');
        } else {
            showToast('📈 Forecast generated successfully!', 'success');
            displayForecast(data);
        }
        
    } catch (err) {
        hideLoading(loadingInterval);
        showToast('❌ Forecast error: ' + err.message, 'error');
    }
    
    if (btn) btn.disabled = false;
}

function displayForecast(data) {
    // Show metrics
    const metricsPanel = document.getElementById('forecastMetrics');
    if (metricsPanel) {
        metricsPanel.style.display = 'grid';
        updateElement('forecastMethod', data.method || 'Holt-Winters');
        const metrics = data.metrics || {};
        updateElement('forecastR2', metrics.R2 ? `${(metrics.R2 * 100).toFixed(1)}%` : 'N/A');
        updateElement('forecastRMSE', metrics.RMSE ? metrics.RMSE.toFixed(2) : 'N/A');
    }
    
    // Chart - API returns forecast as [{date, predicted_kwh}]
    const forecast = data.forecast || [];
    if (forecast.length > 0) {
        const forecastTrace = {
            x: forecast.map(f => f.date),
            y: forecast.map(f => f.predicted_kwh),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Forecast',
            line: { color: '#ff6b6b', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 107, 107, 0.1)'
        };
        
        const layout = getChartLayout('', true);
        Plotly.newPlot('forecastChart', [forecastTrace], layout, chartConfig);
    }
    
    // Table
    const tbody = document.getElementById('forecastTbody');
    const tableCard = document.getElementById('forecastTable');
    
    if (tbody && forecast.length > 0) {
        tbody.innerHTML = forecast.map((f, i) => {
            const val = f.predicted_kwh;
            const prev = i > 0 ? forecast[i-1].predicted_kwh : val;
            const trend = val > prev ? '↑' : val < prev ? '↓' : '→';
            const trendClass = val > prev ? 'up' : val < prev ? 'down' : 'neutral';
            
            return `
                <tr>
                    <td>${f.date}</td>
                    <td>${formatNumber(val, 2)} kWh</td>
                    <td class="trend ${trendClass}">${trend}</td>
                </tr>
            `;
        }).join('');
        
        tableCard.style.display = 'block';
    }
}

async function loadPrediction() {
    const btn = document.getElementById('predictBtn');
    if (btn) btn.disabled = true;
    
    const loadingInterval = showLoading('Loading Prediction', 'Generating 7-day hourly predictions...');
    
    try {
        const response = await fetch('/api/predict');
        const data = await response.json();
        
        hideLoading(loadingInterval);
        
        if (data.status === 'error') {
            showToast(`❌ ${data.message || 'Prediction failed'}`, 'error');
        } else {
            showToast('🤖 Prediction loaded!', 'success');
            displayPrediction(data);
        }
        
    } catch (err) {
        hideLoading(loadingInterval);
        showToast('❌ Prediction error: ' + err.message, 'error');
    }
    
    if (btn) btn.disabled = false;
}

function displayPrediction(data) {
    // The API returns { predictions: [{datetime, predicted_kwh}, ...] }
    const predictions = data.predictions || [];
    
    // Extract timestamps and values
    const timestamps = predictions.map(p => p.datetime);
    const values = predictions.map(p => p.predicted_kwh);
    
    // Chart
    const trace = {
        x: timestamps,
        y: values,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#8a2be2', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(138, 43, 226, 0.1)'
    };
    
    Plotly.newPlot('predictionChart', [trace], getChartLayout(), chartConfig);
    
    // Calculate total and daily totals
    const totalPredicted = values.reduce((sum, v) => sum + v, 0);
    updateElement('predictedUsage', formatNumber(totalPredicted, 1) + ' kWh');
    
    // Group by day for table
    const dailyTotals = {};
    predictions.forEach(p => {
        const date = p.datetime.split(' ')[0];
        if (!dailyTotals[date]) dailyTotals[date] = 0;
        dailyTotals[date] += p.predicted_kwh;
    });
    
    // Table
    const tbody = document.getElementById('predTbody');
    const tableCard = document.getElementById('predictionTable');
    
    const days = Object.entries(dailyTotals);
    if (tbody && days.length > 0) {
        tbody.innerHTML = days.map(([date, total], i) => {
            const prev = i > 0 ? days[i-1][1] : total;
            const trend = total > prev ? '↑' : total < prev ? '↓' : '→';
            const trendClass = total > prev ? 'up' : total < prev ? 'down' : 'neutral';
            
            return `
                <tr>
                    <td>${date}</td>
                    <td>${formatNumber(total, 2)} kWh</td>
                    <td class="trend ${trendClass}">${trend}</td>
                </tr>
            `;
        }).join('');
        
        tableCard.style.display = 'block';
    }
    
    // Add notification
    addNotification('success', '7-Day Prediction Ready', `Total forecast: ${formatNumber(totalPredicted, 1)} kWh`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY DETECTION
// ═══════════════════════════════════════════════════════════════════════════════
async function loadAnomalies() {
    const btn = document.getElementById('anomalyBtn');
    if (btn) btn.disabled = true;
    
    const loadingInterval = showLoading('Detecting Anomalies', 'Running Isolation Forest & Z-Score analysis...');
    
    try {
        const response = await fetch('/api/anomalies');
        const data = await response.json();
        
        hideLoading(loadingInterval);
        
        if (data.status === 'error') {
            showToast(`❌ ${data.message || 'Detection failed'}`, 'error');
        } else {
            showToast(`🔍 Found ${data.total_anomalies || 0} anomalies`, data.total_anomalies > 0 ? 'warning' : 'success');
            displayAnomalies(data);
            
            // Add to notifications if anomalies found
            if (data.total_anomalies > 0) {
                addNotification('warning', 'Anomalies Detected', `Found ${data.total_anomalies} unusual energy patterns`);
            }
        }
        
    } catch (err) {
        hideLoading(loadingInterval);
        showToast('❌ Detection error: ' + err.message, 'error');
    }
    
    if (btn) btn.disabled = false;
}

function displayAnomalies(data) {
    // Stats
    const statsPanel = document.getElementById('anomalyStats');
    if (statsPanel) {
        statsPanel.style.display = 'grid';
        statsPanel.innerHTML = `
            <div class="stat-card glass-panel">
                <div class="stat-icon gradient-red"><i class="fas fa-exclamation-circle"></i></div>
                <div class="stat-info">
                    <span class="stat-value">${data.total_anomalies || 0}</span>
                    <span class="stat-label">Total Anomalies</span>
                </div>
            </div>
            <div class="stat-card glass-panel">
                <div class="stat-icon gradient-orange"><i class="fas fa-database"></i></div>
                <div class="stat-info">
                    <span class="stat-value">${data.total_records || 0}</span>
                    <span class="stat-label">Total Records</span>
                </div>
            </div>
            <div class="stat-card glass-panel">
                <div class="stat-icon gradient-purple"><i class="fas fa-percentage"></i></div>
                <div class="stat-info">
                    <span class="stat-value">${data.pct_anomalies || 0}%</span>
                    <span class="stat-label">Anomaly Rate</span>
                </div>
            </div>
        `;
    }
    
    // Chart - API returns daily_counts as [{date, count}]
    if (data.daily_counts && data.daily_counts.length > 0) {
        const dates = data.daily_counts.map(d => d.date);
        const counts = data.daily_counts.map(d => d.count);
        const colors = counts.map(c => c > 0 ? '#ff6b6b' : 'rgba(100,100,100,0.3)');
        const trace = {
            x: dates,
            y: counts,
            type: 'bar',
            marker: { color: colors }
        };
        Plotly.newPlot('anomalyChart', [trace], getChartLayout(), chartConfig);
    }
    
    // Table - API returns anomalies as [{datetime, energy_kwh, method}]
    const tbody = document.getElementById('anomalyTbody');
    const tableCard = document.getElementById('anomalyTable');
    
    if (tbody && data.anomalies && data.anomalies.length > 0) {
        tbody.innerHTML = data.anomalies.slice(0, 20).map(r => `
            <tr class="anomaly-row">
                <td>${r.datetime}</td>
                <td>${formatNumber(r.energy_kwh, 3)} kWh</td>
                <td><span class="method-badge ${r.method.toLowerCase().replace(/\s/g, '-')}">${r.method}</span></td>
            </tr>
        `).join('');
        tableCard.style.display = 'block';
    }
    
    // Show alert if many anomalies
    if (data.total_anomalies > 10) {
        showAlert(`High anomaly count detected: ${data.total_anomalies} unusual patterns found!`);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THRESHOLD ALERTS
// ═══════════════════════════════════════════════════════════════════════════════
async function loadAlerts() {
    const btn = document.getElementById('alertBtn');
    if (btn) btn.disabled = true;
    
    const dailyThreshold = document.getElementById('dailyThreshold')?.value || '';
    const hourlyThreshold = document.getElementById('hourlyThreshold')?.value || '';
    
    let url = '/api/alerts';
    const params = [];
    if (dailyThreshold) params.push(`daily_threshold=${dailyThreshold}`);
    if (hourlyThreshold) params.push(`hourly_threshold=${hourlyThreshold}`);
    if (params.length > 0) url += '?' + params.join('&');
    
    const loadingInterval = showLoading('Checking Alerts', 'Analyzing threshold violations...');
    
    try {
        const response = await fetch(url);
        const data = await response.json();
        
        hideLoading(loadingInterval);
        displayAlerts(data);
        
    } catch (err) {
        hideLoading(loadingInterval);
        showToast('❌ Alert check error: ' + err.message, 'error');
    }
    
    if (btn) btn.disabled = false;
}

function displayAlerts(data) {
    // Summary - API returns critical_count, warning_count, summary.days_over/hours_over
    const summary = document.getElementById('alertSummary');
    if (summary) {
        summary.style.display = 'grid';
        summary.innerHTML = `
            <div class="stat-card glass-panel">
                <div class="stat-icon gradient-red"><i class="fas fa-bell"></i></div>
                <div class="stat-info">
                    <span class="stat-value">${data.total_alerts || 0}</span>
                    <span class="stat-label">Total Alerts</span>
                </div>
            </div>
            <div class="stat-card glass-panel">
                <div class="stat-icon gradient-orange"><i class="fas fa-exclamation-circle"></i></div>
                <div class="stat-info">
                    <span class="stat-value">${data.critical_count || 0}</span>
                    <span class="stat-label">Critical</span>
                </div>
            </div>
            <div class="stat-card glass-panel">
                <div class="stat-icon gradient-purple"><i class="fas fa-exclamation-triangle"></i></div>
                <div class="stat-info">
                    <span class="stat-value">${data.warning_count || 0}</span>
                    <span class="stat-label">Warning</span>
                </div>
            </div>
        `;
    }
    
    // Table - API alerts have {type, severity, date, value, threshold, message}
    const tbody = document.getElementById('alertTbody');
    const tableCard = document.getElementById('alertTable');
    
    if (tbody && data.alerts && data.alerts.length > 0) {
        tbody.innerHTML = data.alerts.slice(0, 30).map(a => `
            <tr class="alert-row ${a.severity || 'warning'}">
                <td><span class="severity-badge ${a.severity}">${a.severity || 'Warning'}</span></td>
                <td>${a.type}</td>
                <td>${a.date}</td>
                <td>${formatNumber(a.value, 2)} kWh</td>
                <td>${formatNumber(a.threshold, 2)} kWh</td>
            </tr>
        `).join('');
        tableCard.style.display = 'block';
        
        showToast(`⚠️ ${data.total_alerts} threshold violations found`, 'warning');
    } else {
        showToast('✅ No threshold violations', 'success');
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL-TIME UPDATES (SIMULATED)
// ═══════════════════════════════════════════════════════════════════════════════
function startRealTimeUpdates() {
    // Update charts every 30 seconds with slight variations (simulation)
    chartUpdateInterval = setInterval(() => {
        updateLiveCharts();
    }, 30000);
    
    // Update time indicators
    setInterval(updateTimeIndicators, 1000);
}

function updateLiveCharts() {
    // Add slight random variation to simulate live data
    const charts = ['dailyChart', 'hourlyChart'];
    
    charts.forEach(chartId => {
        const el = document.getElementById(chartId);
        if (el && el.data && el.data[0]) {
            const data = el.data[0];
            if (data.y && data.y.length > 0) {
                // Simulate small fluctuation on last point
                const lastIdx = data.y.length - 1;
                const variation = (Math.random() - 0.5) * 0.02 * data.y[lastIdx];
                const newVal = Math.max(0, data.y[lastIdx] + variation);
                
                Plotly.update(chartId, {
                    y: [[...data.y.slice(0, -1), newVal]]
                }, {}, [0]);
            }
        }
    });
}

function updateTimeIndicators() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    // Update any live indicators if present
    document.querySelectorAll('.live-time').forEach(el => {
        el.textContent = timeStr;
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOTIFICATION PANEL
// ═══════════════════════════════════════════════════════════════════════════════
let notifications = [];

function setupNotifications() {
    const btn = document.getElementById('notificationBtn');
    const panel = document.getElementById('notificationPanel');
    const clearBtn = document.getElementById('clearNotifsBtn');
    
    if (btn) {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleNotificationPanel();
        });
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearNotifications();
        });
    }
    
    // Prevent panel clicks from closing it
    if (panel) {
        panel.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }
    
    // Close panel when clicking outside
    document.addEventListener('click', (e) => {
        const panel = document.getElementById('notificationPanel');
        const wrapper = document.querySelector('.notification-wrapper');
        if (panel && wrapper && !wrapper.contains(e.target)) {
            panel.classList.remove('show');
        }
    });
    
    // Load initial notifications
    loadNotifications();
}

function toggleNotificationPanel() {
    const panel = document.getElementById('notificationPanel');
    
    if (panel) {
        panel.classList.toggle('show');
        
        // Mark as read when opened
        if (panel.classList.contains('show')) {
            const badge = document.getElementById('notifBadge');
            if (badge) {
                // Don't hide badge, just mark as read (user can still see count)
            }
        }
    }
}

async function loadNotifications() {
    try {
        // Check for alerts
        const response = await fetch('/api/alerts');
        const data = await response.json();
        
        if (data.status === 'success' && data.alerts && data.alerts.length > 0) {
            notifications = data.alerts.slice(0, 5).map(alert => ({
                type: alert.severity || 'warning',
                title: `${alert.type} Alert`,
                message: `${alert.value.toFixed(2)} kWh exceeded threshold`,
                time: alert.datetime || 'Recently'
            }));
            
            updateNotificationBadge(notifications.length);
            updateNotificationList();
        } else {
            // Add welcome notification if no alerts
            notifications = [{
                type: 'info',
                title: 'Welcome to EnergyAI!',
                message: 'Train the AI model to start getting energy insights.',
                time: 'Just now'
            }];
            updateNotificationBadge(1);
            updateNotificationList();
        }
    } catch (err) {
        console.log('No initial notifications');
        // Add welcome notification on error too
        notifications = [{
            type: 'info',
            title: 'Welcome to EnergyAI!',
            message: 'Train the AI model to start getting energy insights.',
            time: 'Just now'
        }];
        updateNotificationBadge(1);
        updateNotificationList();
    }
}

function addNotification(type, title, message) {
    notifications.unshift({
        type: type,
        title: title,
        message: message,
        time: new Date().toLocaleTimeString()
    });
    
    // Keep only last 10
    notifications = notifications.slice(0, 10);
    
    updateNotificationBadge(notifications.length);
    updateNotificationList();
}

function updateNotificationBadge(count) {
    const badge = document.getElementById('notifBadge');
    if (badge) {
        badge.textContent = count > 9 ? '9+' : count;
        badge.style.display = count > 0 ? 'flex' : 'none';
    }
}

function updateNotificationList() {
    const list = document.getElementById('notifList');
    if (!list) return;
    
    if (notifications.length === 0) {
        list.innerHTML = '<div class="notif-empty"><i class="fas fa-check-circle"></i> No notifications</div>';
        return;
    }
    
    const iconMap = {
        'warning': 'fa-exclamation-triangle',
        'error': 'fa-times-circle',
        'success': 'fa-check-circle',
        'info': 'fa-info-circle'
    };
    
    list.innerHTML = notifications.map(n => `
        <div class="notif-item ${n.type}">
            <div class="notif-icon"><i class="fas ${iconMap[n.type] || iconMap.info}"></i></div>
            <div class="notif-content">
                <strong>${n.title}</strong>
                <p>${n.message}</p>
                <span class="notif-time">${n.time}</span>
            </div>
        </div>
    `).join('');
}

function clearNotifications() {
    notifications = [];
    updateNotificationBadge(0);
    updateNotificationList();
    showToast('Notifications cleared', 'info');
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEATURE IMPORTANCE (separate load)
// ═══════════════════════════════════════════════════════════════════════════════
async function loadFeatureImportance() {
    try {
        const response = await fetch('/api/importance');
        const data = await response.json();
        
        if (data.status === 'success' && data.importance) {
            displayFeatureImportance(data.importance);
        }
    } catch (err) {
        console.error('Error loading feature importance:', err);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    return parseFloat(num).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function updateElement(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.textContent = value;
        // Add pulse animation
        el.classList.add('value-updated');
        setTimeout(() => el.classList.remove('value-updated'), 500);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLEANUP
// ═══════════════════════════════════════════════════════════════════════════════
window.addEventListener('beforeunload', () => {
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// CONSOLE BRANDING
// ═══════════════════════════════════════════════════════════════════════════════
console.log(`
%c⚡ EnergyAI Dashboard %cv1.0
%cAI-Based Smart Energy Monitoring System
%cINT 428 Project

`, 
'color: #00d9ff; font-size: 20px; font-weight: bold;',
'color: #8a2be2; font-size: 14px;',
'color: #00ffa3; font-size: 12px;',
'color: #888; font-size: 11px;'
);

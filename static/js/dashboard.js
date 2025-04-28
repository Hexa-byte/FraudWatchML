/**
 * FraudWatch Dashboard Script
 * Handles dashboard data loading and chart rendering
 */

document.addEventListener('DOMContentLoaded', function() {
    // Default time range
    let timeRange = 30;
    let charts = {};
    
    // Initialize dashboard
    initializeDashboard();
    
    // Event listeners
    document.querySelectorAll('.time-range').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const days = parseInt(this.dataset.days);
            timeRange = days;
            
            // Update active class
            document.querySelectorAll('.time-range').forEach(el => el.classList.remove('active'));
            this.classList.add('active');
            
            // Update dropdown text
            document.getElementById('timeRangeSelector').innerText = `Last ${days} Days`;
            
            // Reload dashboard data
            loadDashboardData();
        });
    });
    
    document.getElementById('refreshDashboard').addEventListener('click', function(e) {
        e.preventDefault();
        loadDashboardData();
    });
    
    document.getElementById('refreshMerchantRisk').addEventListener('click', function(e) {
        e.preventDefault();
        loadMerchantRiskData();
    });
    
    /**
     * Initialize the dashboard
     */
    function initializeDashboard() {
        // Show loading indicator
        showLoading(true);
        
        // Load dashboard data
        loadDashboardData();
    }
    
    /**
     * Load all dashboard data
     */
    function loadDashboardData() {
        showLoading(true);
        
        // Load metrics summary
        fetch(`/monitoring/metrics/summary?days=${timeRange}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                updateSummaryMetrics(data);
                renderDailyTransactionsChart(data.daily_stats);
                renderPaymentMethodChart(data.payment_method_stats);
                renderModelPerformanceChart(data.model_performance);
                loadModelDriftData();
                loadRecentTransactions();
                loadMerchantRiskData();
                showLoading(false);
            })
            .catch(error => {
                console.error('Error loading dashboard data:', error);
                showLoading(false);
                showError('Failed to load dashboard data. Please try again.');
            });
    }
    
    /**
     * Load model drift metrics
     */
    function loadModelDriftData() {
        fetch(`/monitoring/metrics/model-drift?days=${timeRange}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                renderModelDriftChart(data.drift_metrics);
            })
            .catch(error => {
                console.error('Error loading model drift data:', error);
            });
    }
    
    /**
     * Load recent transactions
     */
    function loadRecentTransactions() {
        fetch('/api/v1/transactions/recent?limit=5&fraud_only=true')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                updateRecentTransactionsTable(data.transactions);
            })
            .catch(error => {
                console.error('Error loading recent transactions:', error);
            });
    }
    
    /**
     * Load merchant risk data
     */
    function loadMerchantRiskData() {
        fetch(`/monitoring/transactions/merchant-risk?days=${timeRange}&limit=5`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                updateMerchantRiskTable(data.merchant_risk_metrics);
            })
            .catch(error => {
                console.error('Error loading merchant risk data:', error);
            });
    }
    
    /**
     * Update summary metrics on the dashboard
     */
    function updateSummaryMetrics(data) {
        const summary = data.summary;
        const model = data.model;
        
        // Update summary cards
        document.getElementById('totalTransactions').textContent = summary.total_transactions.toLocaleString();
        document.getElementById('fraudRate').textContent = summary.fraud_rate.toFixed(2) + '%';
        document.getElementById('falsePositiveRate').textContent = summary.false_positive_rate.toFixed(2) + '%';
        document.getElementById('modelVersion').textContent = model.version || 'Not Available';
    }
    
    /**
     * Update recent transactions table
     */
    function updateRecentTransactionsTable(transactions) {
        const tableBody = document.getElementById('recentTransactionsTable');
        
        // Clear table
        tableBody.innerHTML = '';
        
        if (transactions.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="7" class="text-center">No recent fraud transactions found</td>';
            tableBody.appendChild(row);
            return;
        }
        
        // Add transactions to table
        transactions.forEach(tx => {
            const row = document.createElement('tr');
            
            // Format date
            const txDate = new Date(tx.timestamp);
            const formattedDate = txDate.toLocaleString();
            
            // Format amount
            const formattedAmount = parseFloat(tx.amount).toLocaleString('en-US', {
                style: 'currency',
                currency: 'USD'
            });
            
            // Generate row HTML
            row.innerHTML = `
                <td>${tx.id}</td>
                <td>${formattedDate}</td>
                <td>${formattedAmount}</td>
                <td>${tx.customer_id}</td>
                <td>${tx.fraud_score.toFixed(2)}</td>
                <td>
                    ${tx.reviewed ? 
                        (tx.review_result ? 
                            '<span class="badge bg-danger">Confirmed Fraud</span>' : 
                            '<span class="badge bg-success">Legitimate</span>'
                        ) : 
                        '<span class="badge bg-warning">Pending Review</span>'
                    }
                </td>
                <td>
                    <a href="/transactions/${tx.id}" class="btn btn-sm btn-primary">
                        <i class="fas fa-eye"></i> Details
                    </a>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    /**
     * Update merchant risk table
     */
    function updateMerchantRiskTable(merchants) {
        const tableBody = document.getElementById('merchantRiskTable');
        
        // Clear table
        tableBody.innerHTML = '';
        
        if (!merchants || merchants.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="4" class="text-center">No merchant risk data available</td>';
            tableBody.appendChild(row);
            return;
        }
        
        // Add merchants to table
        merchants.forEach(merchant => {
            const row = document.createElement('tr');
            
            // Format amount
            const formattedAmount = parseFloat(merchant.avg_amount).toLocaleString('en-US', {
                style: 'currency',
                currency: 'USD'
            });
            
            // Generate row HTML
            row.innerHTML = `
                <td>${merchant.merchant_id}</td>
                <td>${merchant.tx_count.toLocaleString()}</td>
                <td><span class="badge ${merchant.fraud_rate > 5 ? 'bg-danger' : 'bg-warning'}">${merchant.fraud_rate.toFixed(2)}%</span></td>
                <td>${formattedAmount}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    /**
     * Render daily transactions chart
     */
    function renderDailyTransactionsChart(dailyStats) {
        const ctx = document.getElementById('dailyTransactionsChart').getContext('2d');
        
        // Prepare data
        const labels = [];
        const txCounts = [];
        const fraudCounts = [];
        const fraudRates = [];
        
        dailyStats.forEach(day => {
            // Format date as MM/DD
            const date = new Date(day.date);
            const formattedDate = `${date.getMonth() + 1}/${date.getDate()}`;
            
            labels.push(formattedDate);
            txCounts.push(day.tx_count);
            fraudCounts.push(day.fraud_count);
            fraudRates.push(day.fraud_rate);
        });
        
        // Destroy previous chart if it exists
        if (charts.dailyTransactions) {
            charts.dailyTransactions.destroy();
        }
        
        // Create new chart
        charts.dailyTransactions = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Total Transactions',
                        data: txCounts,
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Fraud Transactions',
                        data: fraudCounts,
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Fraud Rate (%)',
                        data: fraudRates,
                        type: 'line',
                        fill: false,
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Transaction Count'
                        }
                    },
                    y1: {
                        position: 'right',
                        beginAtZero: true,
                        max: 10,
                        title: {
                            display: true,
                            text: 'Fraud Rate (%)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.dataset.yAxisID === 'y1') {
                                    label += context.parsed.y.toFixed(2) + '%';
                                } else {
                                    label += context.parsed.y.toLocaleString();
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Render payment method chart
     */
    function renderPaymentMethodChart(paymentStats) {
        const ctx = document.getElementById('paymentMethodChart').getContext('2d');
        
        // Prepare data
        const labels = [];
        const txCounts = [];
        const fraudRates = [];
        
        paymentStats.forEach(method => {
            labels.push(method.payment_method);
            txCounts.push(method.tx_count);
            fraudRates.push(method.fraud_rate);
        });
        
        // Destroy previous chart if it exists
        if (charts.paymentMethod) {
            charts.paymentMethod.destroy();
        }
        
        // Create new chart
        charts.paymentMethod = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [
                    {
                        data: txCounts,
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.7)',
                            'rgba(255, 205, 86, 0.7)',
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(153, 102, 255, 0.7)',
                            'rgba(201, 203, 207, 0.7)'
                        ],
                        borderColor: [
                            'rgba(75, 192, 192, 1)',
                            'rgba(255, 205, 86, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(201, 203, 207, 1)'
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.formattedValue;
                                const dataset = context.dataset;
                                const total = dataset.data.reduce((acc, data) => acc + data, 0);
                                const percentage = Math.round((context.raw / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            },
                            afterLabel: function(context) {
                                const index = context.dataIndex;
                                return `Fraud Rate: ${fraudRates[index].toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Render model performance chart
     */
    function renderModelPerformanceChart(performanceData) {
        const ctx = document.getElementById('modelPerformanceChart').getContext('2d');
        
        if (!performanceData || performanceData.length === 0) {
            // If no performance data, show message
            if (charts.modelPerformance) {
                charts.modelPerformance.destroy();
            }
            return;
        }
        
        // Prepare data
        const labels = [];
        const falsePositives = [];
        const falseNegatives = [];
        const latency = [];
        
        performanceData.forEach(day => {
            // Format date as MM/DD
            const date = new Date(day.date);
            const formattedDate = `${date.getMonth() + 1}/${date.getDate()}`;
            
            labels.push(formattedDate);
            falsePositives.push(day.false_positives);
            falseNegatives.push(day.false_negatives);
            latency.push(day.avg_latency_ms);
        });
        
        // Destroy previous chart if it exists
        if (charts.modelPerformance) {
            charts.modelPerformance.destroy();
        }
        
        // Create new chart
        charts.modelPerformance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'False Positives',
                        data: falsePositives,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'False Negatives',
                        data: falseNegatives,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Avg Latency (ms)',
                        data: latency,
                        borderColor: 'rgba(255, 205, 86, 1)',
                        backgroundColor: 'rgba(255, 205, 86, 0.1)',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Error Count'
                        }
                    },
                    y1: {
                        position: 'right',
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Latency (ms)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Render model drift chart
     */
    function renderModelDriftChart(driftData) {
        const ctx = document.getElementById('modelDriftChart').getContext('2d');
        
        if (!driftData || driftData.length === 0) {
            // If no drift data, show message
            if (charts.modelDrift) {
                charts.modelDrift.destroy();
            }
            return;
        }
        
        // Prepare data
        const labels = [];
        const klDivergence = [];
        
        driftData.forEach(day => {
            // Format date as MM/DD
            const date = new Date(day.date);
            const formattedDate = `${date.getMonth() + 1}/${date.getDate()}`;
            
            labels.push(formattedDate);
            klDivergence.push(day.kl_divergence);
        });
        
        // Destroy previous chart if it exists
        if (charts.modelDrift) {
            charts.modelDrift.destroy();
        }
        
        // Create new chart
        charts.modelDrift = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'KL Divergence',
                        data: klDivergence,
                        borderColor: 'rgba(153, 102, 255, 1)',
                        backgroundColor: 'rgba(153, 102, 255, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'KL Divergence'
                        }
                    }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            thresholdLine: {
                                type: 'line',
                                yMin: 0.1,
                                yMax: 0.1,
                                borderColor: 'rgba(255, 99, 132, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    enabled: true,
                                    content: 'Drift Threshold',
                                    position: 'end'
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Show or hide loading indicator
     */
    function showLoading(isLoading) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        const dashboardContent = document.getElementById('dashboardContent');
        
        if (isLoading) {
            loadingIndicator.style.display = 'block';
            dashboardContent.style.opacity = '0.5';
        } else {
            loadingIndicator.style.display = 'none';
            dashboardContent.style.opacity = '1';
        }
    }
    
    /**
     * Show error message
     */
    function showError(message) {
        // Create alert element
        const alertElement = document.createElement('div');
        alertElement.className = 'alert alert-danger alert-dismissible fade show';
        alertElement.role = 'alert';
        alertElement.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert at top of container
        const container = document.querySelector('main.container');
        container.insertBefore(alertElement, container.firstChild);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            alertElement.classList.remove('show');
            setTimeout(() => alertElement.remove(), 150);
        }, 5000);
    }
});

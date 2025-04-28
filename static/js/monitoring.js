/**
 * FraudWatch Monitoring Script
 * Specialized monitoring functions beyond the basic dashboard
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize datepickers if present
    if (document.getElementById('start_date')) {
        // Set default date range (last 7 days)
        const today = new Date();
        const lastWeek = new Date(today);
        lastWeek.setDate(lastWeek.getDate() - 7);
        
        document.getElementById('start_date').valueAsDate = lastWeek;
        document.getElementById('end_date').valueAsDate = today;
    }
    
    // Initialize high value transaction threshold slider if present
    const thresholdSlider = document.getElementById('thresholdSlider');
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', function() {
            document.getElementById('thresholdValue').textContent = '$' + parseInt(this.value).toLocaleString();
        });
        
        document.getElementById('highValueForm').addEventListener('submit', function(e) {
            e.preventDefault();
            loadHighValueMetrics(thresholdSlider.value);
        });
        
        // Initial load
        loadHighValueMetrics(thresholdSlider.value);
    }
    
    // Real-time transaction monitoring if present
    const realtimeMonitor = document.getElementById('realtimeMonitor');
    if (realtimeMonitor) {
        initializeRealtimeMonitoring();
    }
    
    // Alert configuration testing if present
    const testAlertButton = document.getElementById('testAlertCondition');
    if (testAlertButton) {
        testAlertButton.addEventListener('click', function() {
            testAlertCondition();
        });
    }
    
    /**
     * Load high-value transaction metrics
     */
    function loadHighValueMetrics(threshold) {
        const days = document.getElementById('timeRange').value;
        
        fetch(`/monitoring/metrics/high-value?days=${days}&threshold=${threshold}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                updateHighValueMetrics(data);
            })
            .catch(error => {
                console.error('Error loading high-value metrics:', error);
                showError('Failed to load high-value transaction metrics.');
            });
    }
    
    /**
     * Update high-value transaction metrics UI
     */
    function updateHighValueMetrics(data) {
        // Update summary numbers
        document.getElementById('totalHighValue').textContent = data.total_count.toLocaleString();
        document.getElementById('flaggedHighValue').textContent = data.flagged_count.toLocaleString();
        document.getElementById('highValueRate').textContent = data.flag_rate.toFixed(2) + '%';
        
        // Render chart if container exists
        const chartContainer = document.getElementById('highValueChart');
        if (chartContainer) {
            renderHighValueChart(chartContainer, data.daily_breakdown);
        }
        
        // Update table if it exists
        const tableBody = document.getElementById('highValueTableBody');
        if (tableBody) {
            updateHighValueTable(tableBody, data.daily_breakdown);
        }
    }
    
    /**
     * Render high-value transactions chart
     */
    function renderHighValueChart(container, dailyData) {
        // Prepare data
        const labels = [];
        const counts = [];
        const flaggedCounts = [];
        const rates = [];
        
        dailyData.forEach(day => {
            // Format date as MM/DD
            const date = new Date(day.date);
            const formattedDate = `${date.getMonth() + 1}/${date.getDate()}`;
            
            labels.push(formattedDate);
            counts.push(day.count);
            flaggedCounts.push(day.flagged);
            rates.push(day.rate);
        });
        
        // Create chart
        const ctx = container.getContext('2d');
        
        // Destroy previous chart if it exists
        if (window.highValueChart) {
            window.highValueChart.destroy();
        }
        
        window.highValueChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'High-Value Transactions',
                        data: counts,
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Flagged Transactions',
                        data: flaggedCounts,
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Flag Rate (%)',
                        data: rates,
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
                        max: 30,
                        title: {
                            display: true,
                            text: 'Flag Rate (%)'
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
     * Update high-value transactions table
     */
    function updateHighValueTable(tableBody, dailyData) {
        // Clear table
        tableBody.innerHTML = '';
        
        if (dailyData.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="4" class="text-center">No high-value transaction data available</td>';
            tableBody.appendChild(row);
            return;
        }
        
        // Add data to table
        dailyData.forEach(day => {
            const row = document.createElement('tr');
            
            // Format date
            const date = new Date(day.date);
            const formattedDate = date.toLocaleDateString();
            
            // Generate row HTML
            row.innerHTML = `
                <td>${formattedDate}</td>
                <td>${day.count.toLocaleString()}</td>
                <td>${day.flagged.toLocaleString()}</td>
                <td>${day.rate.toFixed(2)}%</td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    /**
     * Initialize real-time transaction monitoring
     */
    function initializeRealtimeMonitoring() {
        // In a real implementation, this would use WebSockets or Server-Sent Events
        // to receive real-time transaction updates. For this demo, we'll simulate
        // using a polling mechanism.
        
        const transactionList = document.getElementById('realtimeTransactions');
        const statusIndicator = document.getElementById('connectionStatus');
        
        // Set connection status
        statusIndicator.className = 'badge bg-success';
        statusIndicator.textContent = 'Connected';
        
        // Check for new transactions every 10 seconds
        const intervalId = setInterval(fetchLatestTransactions, 10000);
        
        // Initial fetch
        fetchLatestTransactions();
        
        // Store interval ID for cleanup
        window.realtimeMonitoringInterval = intervalId;
        
        // Fetch latest transactions
        function fetchLatestTransactions() {
            // In a real implementation, this would fetch only new transactions
            // since the last check. For this demo, we'll just get the most recent ones.
            fetch('/api/v1/transactions/recent?limit=10')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    updateRealtimeTransactions(transactionList, data.transactions);
                })
                .catch(error => {
                    console.error('Error fetching latest transactions:', error);
                    statusIndicator.className = 'badge bg-danger';
                    statusIndicator.textContent = 'Disconnected';
                });
        }
    }
    
    /**
     * Update real-time transaction list
     */
    function updateRealtimeTransactions(listElement, transactions) {
        // Don't clear the list, just prepend new transactions
        // In a real implementation, we'd check for duplicates
        
        if (!transactions || transactions.length === 0) {
            return;
        }
        
        const fragment = document.createDocumentFragment();
        const existingIds = Array.from(listElement.children).map(item => 
            item.dataset.txId
        );
        
        transactions.forEach(tx => {
            // Skip if already in list
            if (existingIds.includes(tx.id)) {
                return;
            }
            
            const listItem = document.createElement('div');
            listItem.className = 'list-group-item list-group-item-action';
            listItem.dataset.txId = tx.id;
            
            // Format date
            const txDate = new Date(tx.timestamp);
            const formattedDate = txDate.toLocaleString();
            
            // Format amount
            const formattedAmount = parseFloat(tx.amount).toLocaleString('en-US', {
                style: 'currency',
                currency: 'USD'
            });
            
            // Risk class based on fraud score
            let riskClass = '';
            if (tx.fraud_score > 0.7) {
                riskClass = 'text-danger';
            } else if (tx.fraud_score > 0.5) {
                riskClass = 'text-warning';
            }
            
            // Generate HTML
            listItem.innerHTML = `
                <div class="d-flex w-100 justify-content-between">
                    <h5 class="mb-1 ${riskClass}">${formattedAmount}</h5>
                    <small>${formattedDate}</small>
                </div>
                <p class="mb-1">Transaction ID: ${tx.id}</p>
                <div class="d-flex justify-content-between">
                    <small>Customer: ${tx.customer_id}</small>
                    <span class="badge ${tx.is_fraud ? 'bg-danger' : 'bg-success'}">
                        ${tx.is_fraud ? 'Flagged' : 'Passed'}
                    </span>
                </div>
            `;
            
            fragment.prepend(listItem);
        });
        
        // Add new transactions to the top of the list
        listElement.prepend(fragment);
        
        // Limit list to 20 items
        while (listElement.children.length > 20) {
            listElement.removeChild(listElement.lastChild);
        }
    }
    
    /**
     * Test alert condition
     */
    function testAlertCondition() {
        const conditionInput = document.getElementById('condition');
        const thresholdInput = document.getElementById('threshold');
        const resultContainer = document.getElementById('alertTestResult');
        
        if (!conditionInput || !thresholdInput || !resultContainer) {
            return;
        }
        
        try {
            // Parse condition JSON
            const condition = JSON.parse(conditionInput.value);
            const threshold = parseFloat(thresholdInput.value);
            
            // Validate condition
            if (!condition.field || !condition.operator) {
                throw new Error('Invalid condition format. Requires "field" and "operator" properties.');
            }
            
            // Create sample transaction
            const sampleTransaction = {
                amount: 5000,
                fraud_score: 0.75,
                payment_method: 'card',
                card_present: false
            };
            
            // Set the specified field to be just above the threshold
            if (condition.field === 'amount') {
                sampleTransaction.amount = threshold + 1;
            } else if (condition.field === 'fraud_score') {
                sampleTransaction.fraud_score = Math.min(threshold + 0.01, 1.0);
            }
            
            // Evaluate condition
            let conditionMet = false;
            const fieldValue = sampleTransaction[condition.field];
            
            switch(condition.operator) {
                case 'gt':
                    conditionMet = fieldValue > threshold;
                    break;
                case 'lt':
                    conditionMet = fieldValue < threshold;
                    break;
                case 'eq':
                    conditionMet = fieldValue === threshold;
                    break;
                case 'neq':
                    conditionMet = fieldValue !== threshold;
                    break;
                case 'in':
                    if (Array.isArray(condition.value)) {
                        conditionMet = condition.value.includes(fieldValue);
                    }
                    break;
                default:
                    throw new Error(`Unknown operator: ${condition.operator}`);
            }
            
            // Display result
            resultContainer.className = conditionMet ? 'alert alert-success' : 'alert alert-danger';
            resultContainer.innerHTML = conditionMet
                ? '<i class="fas fa-check-circle me-2"></i>Condition triggered successfully!'
                : '<i class="fas fa-times-circle me-2"></i>Condition not triggered with test values.';
            
            // Show sample transaction
            resultContainer.innerHTML += `
                <div class="mt-2">
                    <strong>Sample transaction:</strong>
                    <pre class="mt-1 p-2 bg-dark">${JSON.stringify(sampleTransaction, null, 2)}</pre>
                </div>
            `;
            
        } catch (error) {
            resultContainer.className = 'alert alert-danger';
            resultContainer.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Error: ${error.message}`;
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

import React, { useState, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import './App.css'; // Import the CSS file

// --- Mock Data Generator (Simulating an API) ---
const generateMockData = (count) => {
  const products = [];
  const categories = ["Electronics", "Home", "Automotive", "Fashion", "Office"];
  
  for (let i = 1; i <= count; i++) {
    const avgSales = Math.floor(Math.random() * 50) + 5;
    const leadTime = Math.floor(Math.random() * 14) + 1;
    const dailySales = avgSales / 7;
    const demandDuringLead = dailySales * leadTime;
    const safetyStock = demandDuringLead * 0.5;
    const reorderPoint = Math.ceil(demandDuringLead + safetyStock);
    
    const isLowStock = Math.random() > 0.6; 
    let currentInventory;
    
    if (isLowStock) {
      currentInventory = Math.floor(Math.random() * reorderPoint); 
    } else {
      currentInventory = Math.floor(reorderPoint + (Math.random() * 100));
    }

    products.push({
      id: i,
      name: `${categories[i % categories.length]} Item ${i}`,
      inventory: currentInventory,
      avgSales: avgSales,
      leadTime: leadTime,
      actualNeedReorder: currentInventory <= reorderPoint ? 1 : 0
    });
  }
  return products;
};

export default function App() {
  const [products, setProducts] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState("Idle");
  const [accuracy, setAccuracy] = useState(null);

  // 1. Fetch Data
  const fetchData = useCallback(async () => {
    setLoading(true);
    setModelStatus("Fetching Data...");
    setTimeout(() => {
      const data = generateMockData(100);
      setProducts(data);
      setPredictions({});
      setAccuracy(null);
      setLoading(false);
      setModelStatus("Data Loaded. Ready to Train.");
    }, 1000);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // 2. Train Model & Predict
  const runPredictionModel = async () => {
    if (products.length === 0) return;

    setLoading(true);
    setModelStatus("Preparing Tensors...");

    const inputs = products.map(p => [p.inventory, p.avgSales, p.leadTime]);
    const labels = products.map(p => p.actualNeedReorder);

    const inputTensor = tf.tensor2d(inputs);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max(0);
    const inputMin = inputTensor.min(0);
    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
      optimizer: tf.train.adam(0.05),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    setModelStatus("Training Model (50 Epochs)...");

    await model.fit(normalizedInputs, labelTensor, {
      epochs: 50,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 10 === 0) console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
        }
      }
    });

    const evalResult = model.evaluate(normalizedInputs, labelTensor);
    const acc = evalResult[1].dataSync()[0];
    setAccuracy((acc * 100).toFixed(1));

    setModelStatus("Running Predictions...");
    
    const predictionResult = model.predict(normalizedInputs);
    const predictionData = await predictionResult.data();

    const newPredictions = {};
    predictionData.forEach((val, index) => {
      newPredictions[products[index].id] = val > 0.5 ? "Reorder" : "Healthy";
    });

    setPredictions(newPredictions);
    setModelStatus("Complete");
    setLoading(false);

    tf.dispose([inputTensor, labelTensor, normalizedInputs, inputMax, inputMin, predictionResult]);
  };

  const reorderCount = Object.values(predictions).filter(p => p === "Reorder").length;

  return (
    <div className="forecast-app">
      <div className="container">
        {/* Header */}
        <div className="header">
          <div className="title">
            <h1><span>📦</span> Forecast AI</h1>
            <p>TensorFlow.js Inventory Predictor</p>
          </div>
          
          <div className="controls">
            <button onClick={fetchData} disabled={loading}>
              <span>🔄</span> Generate Data
            </button>
            <button className="primary" onClick={runPredictionModel} disabled={loading || products.length === 0}>
              {loading ? <span>⏳</span> : <span>📈</span>}
              Train & Predict
            </button>
          </div>
        </div>

        {/* Status Bar */}
        <div className="stats-grid">
          <StatCard title="Total Products" value={products.length} icon="📦" />
          <StatCard title="Model Status" value={modelStatus} icon="⏱️" highlight={loading} />
          <StatCard title="Predicted Reorders" value={Object.keys(predictions).length > 0 ? reorderCount : "-"} icon="⚠️" isWarning />
          <StatCard title="Model Accuracy" value={accuracy ? `${accuracy}%` : "-"} icon="✅" />
        </div>

        {/* Main Data Table */}
        <div className="table-container">
          <div className="table-header">
            <h3>Inventory Analysis</h3>
            <span className="badge-count">Showing {products.length} items</span>
          </div>
          
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Product Name</th>
                  <th>Current Stock</th>
                  <th>Avg. Sales/Week</th>
                  <th>Lead Time (Days)</th>
                  <th>AI Prediction</th>
                </tr>
              </thead>
              <tbody>
                {products.length === 0 ? (
                  <tr>
                    <td colSpan="5" style={{ textAlign: "center", padding: "40px", color: "#6c757d" }}>
                      No data loaded. Click "Generate Data" to begin.
                    </td>
                  </tr>
                ) : (
                  products.map((product) => {
                    const status = predictions[product.id];
                    return (
                      <tr key={product.id}>
                        <td style={{ fontWeight: 600, color: '#212529' }}>{product.name}</td>
                        <td>
                          <span className={`stock-tag ${product.inventory < 20 ? 'low' : 'ok'}`}>
                            {product.inventory} units
                          </span>
                        </td>
                        <td style={{ color: '#6c757d' }}>{product.avgSales}</td>
                        <td style={{ color: '#6c757d' }}>{product.leadTime} days</td>
                        <td>
                          {status ? (
                            <span className={`status-badge ${status === "Reorder" ? "reorder" : "healthy"}`}>
                              <span>{status === "Reorder" ? "⚠️" : "✅"}</span>
                              {status}
                            </span>
                          ) : (
                            <span className="loading-text">Pending...</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

// Simple Stat Card Component
function StatCard({ title, value, icon, isWarning = false, highlight = false }) {
  return (
    <div className="stat-card">
      <div className="stat-content">
        <h4>{title}</h4>
        <p className={`value ${isWarning ? "text-warn" : ""}`} style={{ opacity: highlight ? 0.5 : 1 }}>
          {value}
        </p>
      </div>
      <div className="stat-icon">{icon}</div>
    </div>
  );
} 
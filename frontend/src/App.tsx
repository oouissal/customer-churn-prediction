import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import BatchPredictionPage from "./pages/BatchPredictionPage";
import DashboardPage from "./pages/DashboardPage";
import ModelInfoPage from "./pages/ModelInfoPage";
import PredictionPage from "./pages/PredictionPage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/predict" element={<PredictionPage />} />
        <Route path="/model" element={<ModelInfoPage />} />
        <Route path="/batch" element={<BatchPredictionPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Layout>
  );
}

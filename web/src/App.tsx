import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ModelProvider } from "./state/ModelContext";
import { Layout } from "./components/Layout";
import { CompleterPage } from "./pages/CompleterPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import { MetaPage } from "./pages/MetaPage";
import { SciencePage } from "./pages/SciencePage";

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "")}>
      <ModelProvider>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/completer" replace />} />
            <Route path="completer" element={<CompleterPage />} />
            <Route path="analysis" element={<AnalysisPage />} />
            <Route path="meta" element={<MetaPage />} />
            <Route path="science" element={<SciencePage />} />
            <Route path="*" element={<Navigate to="/completer" replace />} />
          </Route>
        </Routes>
      </ModelProvider>
    </BrowserRouter>
  );
}

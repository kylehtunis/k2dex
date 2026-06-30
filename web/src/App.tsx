import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ModelProvider } from "./state/ModelContext";
import { PageStateProvider } from "./state/PageStateContext";
import { FeatureModalProvider } from "./components/FeatureModal";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { CompleterPage } from "./pages/CompleterPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import { MetaPage } from "./pages/MetaPage";

const SciencePage = lazy(() => import("./pages/SciencePage"));

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "")}>
      <ModelProvider>
        <PageStateProvider>
          <FeatureModalProvider>
          <Suspense fallback={null}>
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<HomePage />} />
                <Route path="completer" element={<CompleterPage />} />
                <Route path="analysis" element={<AnalysisPage />} />
                <Route path="meta" element={<MetaPage />} />
                <Route path="science" element={<SciencePage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </Suspense>
          </FeatureModalProvider>
        </PageStateProvider>
      </ModelProvider>
    </BrowserRouter>
  );
}

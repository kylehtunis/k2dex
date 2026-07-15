import { BrowserRouter } from "react-router-dom";
import { ModelProvider } from "./state/ModelContext";
import { PageStateProvider } from "./state/PageStateContext";
import { FeatureModalProvider } from "./components/FeatureModal";
import { AppRoutes } from "./AppRoutes";

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "")}>
      <ModelProvider>
        <PageStateProvider>
          <FeatureModalProvider>
            <AppRoutes />
          </FeatureModalProvider>
        </PageStateProvider>
      </ModelProvider>
    </BrowserRouter>
  );
}

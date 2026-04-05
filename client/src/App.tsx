import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { ControlRibbon } from './components/layout/ControlRibbon';
import { useTrainingState } from './hooks/useTrainingState';
import { OverviewView } from './components/views/OverviewView';
import { ConvergenceView } from './components/views/ConvergenceView';
import { PrivacyView } from './components/views/PrivacyView';
import { CommunicationView } from './components/views/CommunicationView';
import { ComparisonView } from './components/views/ComparisonView';

function AppLayout() {
  const { connected, updates, status, startTraining, stopTraining } = useTrainingState();

  return (
    <div className="flex h-screen overflow-hidden bg-slate-900 text-slate-100">
      {/* Left sidebar */}
      <Sidebar />

      {/* Main content area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Top header */}
        <Header connected={connected} status={status} />

        {/* Control ribbon */}
        <ControlRibbon
          status={status}
          onStart={startTraining}
          onStop={stopTraining}
        />

        {/* Page content */}
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<OverviewView updates={updates} status={status} />} />
            <Route path="/convergence" element={<ConvergenceView updates={updates} />} />
            <Route path="/privacy" element={<PrivacyView updates={updates} />} />
            <Route path="/communication" element={<CommunicationView updates={updates} />} />
            <Route path="/comparison" element={<ComparisonView />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}

export default App;

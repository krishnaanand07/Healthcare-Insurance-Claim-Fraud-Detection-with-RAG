import { useState } from 'react';
import { Sidebar } from './components/Layout/Sidebar';
import { TopBar } from './components/Layout/TopBar';
import { ClaimForm } from './components/ClaimForm/ClaimForm';
import { AnalysisCard } from './components/Analysis/AnalysisCard';
import { AIInvestigation } from './components/AIInvestigation/AIInvestigation';

function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState<any>(null);
  const [investigationResult, setInvestigationResult] = useState<any>(null);
  const [currentClaimPayload, setCurrentClaimPayload] = useState<any>(null);

  const handleAnalyzeClaim = async (payload: any) => {
    setIsAnalyzing(true);
    setEvaluationResult(null);
    setInvestigationResult(null);
    setCurrentClaimPayload(payload);

    try {
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      // 1. Call existing Prediction API (preserved functionality)
      const predResponse = await fetch(`${API_BASE_URL}/api/predict/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (predResponse.ok) {
        const predData = await predResponse.json();
        setEvaluationResult(predData);
      }

      // 2. Call new RAG + NVIDIA LLM AI Investigation API
      const aiResponse = await fetch(`${API_BASE_URL}/api/ai/investigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim: payload })
      });

      if (aiResponse.ok) {
        const aiData = await aiResponse.json();
        setInvestigationResult(aiData);
      } else {
        console.warn('AI Investigation API responded with error status:', aiResponse.status);
      }
    } catch (err: any) {
      console.error('Error in claim analysis:', err);
      alert(err.message || 'Error evaluating claim.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full bg-[#EEF3FA] text-[#172554] font-sans selection:bg-[#FACC15] selection:text-[#172554]">
      {/* Left Sidebar */}
      <div className="hidden md:block">
        <Sidebar />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-h-screen max-w-[100vw] overflow-x-hidden">
        <TopBar />

        {/* Hero Section */}
        <div className="px-8 py-6 shrink-0">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-bold text-[#F59E0B] tracking-widest uppercase mb-2">AI-Powered Claim Verification & Investigation</p>
              <h1 className="text-4xl font-extrabold text-[#172554] tracking-tight mb-3">
                Analyze. Verify. Ensure <span className="text-[#F59E0B]">Fairness.</span>
              </h1>
              <p className="text-[#64748B] text-sm max-w-2xl font-medium leading-relaxed">
                Integrated Healthcare Insurance Claim Fraud Detection System with RAG + NVIDIA LLM decision support.
              </p>
            </div>
            <div className="hidden lg:block bg-white/40 border border-white/60 p-4 rounded-xl shadow-sm backdrop-blur-sm">
              <p className="text-sm italic font-medium text-[#52658F]">
                "Fair claims build stronger communities."
              </p>
            </div>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="flex-1 px-8 pb-12 space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            {/* Form Column */}
            <div className="w-full relative z-10">
              <ClaimForm onSubmit={handleAnalyzeClaim} isAnalyzing={isAnalyzing} />
            </div>

            {/* Analysis Column */}
            <div className="w-full relative z-10">
              <AnalysisCard result={evaluationResult} isLoading={isAnalyzing} />
            </div>
          </div>

          {/* AI Investigation Section (RAG + NVIDIA LLM Report & Chat) */}
          {(investigationResult || isAnalyzing) && (
            <div className="w-full relative z-10">
              <AIInvestigation
                data={investigationResult}
                claimPayload={currentClaimPayload}
                isLoading={isAnalyzing}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

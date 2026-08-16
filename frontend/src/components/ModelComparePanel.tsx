import { useState, useEffect } from 'react';
import type { TrainedModel } from '../types';
import { models as modelApi } from '../api/endpoints';

interface Props {
  availableModels: TrainedModel[];
  onClose: () => void;
}

export default function ModelComparePanel({ availableModels, onClose }: Props) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [comparedModels, setComparedModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(false);

  // Auto-select completed models (up to 5)
  useEffect(() => {
    const completed = availableModels
      .filter(m => m.status === 'completed' && m.metrics && Object.keys(m.metrics).length > 0)
      .slice(0, 5);
    setSelectedIds(new Set(completed.map(m => m.id)));
  }, [availableModels]);

  function toggle(id: string) {
    setSelectedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  async function handleCompare() {
    if (selectedIds.size < 2) return;
    setLoading(true);
    try {
      const res: any = await modelApi.compare(Array.from(selectedIds));
      setComparedModels(res.models || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  // Collect all unique metric keys
  const allMetricKeys = new Set<string>();
  for (const m of comparedModels) {
    if (m.metrics) Object.keys(m.metrics).forEach(k => allMetricKeys.add(k));
  }
  const metricKeys = Array.from(allMetricKeys);

  // Find best value for each metric
  const bestValues: Record<string, { value: number; modelId: string }> = {};
  for (const key of metricKeys) {
    let best = -Infinity;
    let bestId = '';
    for (const m of comparedModels) {
      const v = m.metrics?.[key];
      if (typeof v === 'number' && v > best) { best = v; bestId = m.id; }
    }
    bestValues[key] = { value: best, modelId: bestId };
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-800">📊 模型对比</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-lg leading-none">&times;</button>
      </div>

      {/* Model selection */}
      <div className="mb-4 max-h-40 overflow-y-auto">
        <p className="text-xs text-gray-500 mb-2">选择要对比的模型（至少 2 个）：</p>
        {availableModels.filter(m => m.status === 'completed').length === 0 && (
          <p className="text-xs text-gray-400">暂无已完成的模型</p>
        )}
        {availableModels.filter(m => m.status === 'completed').map(m => (
          <label key={m.id}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm mb-1 transition-all
              ${selectedIds.has(m.id) ? 'bg-violet-50 border border-violet-200' : 'hover:bg-gray-50 border border-transparent'}`}>
            <input type="checkbox" checked={selectedIds.has(m.id)} onChange={() => toggle(m.id)}
              className="w-3.5 h-3.5 rounded text-violet-600" />
            <span className="text-gray-700 truncate flex-1">{m.name}</span>
            {m.metrics && (
              <span className="text-[10px] text-gray-400 font-mono">
                mAP50: {typeof m.metrics.mAP50 === 'number' ? m.metrics.mAP50.toFixed(3) : '-'}
              </span>
            )}
          </label>
        ))}
      </div>

      <button onClick={handleCompare} disabled={selectedIds.size < 2 || loading}
        className="w-full px-4 py-2 text-sm rounded-lg bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 font-medium mb-4">
        {loading ? '加载中...' : `对比 ${selectedIds.size} 个模型`}
      </button>

      {/* Comparison table */}
      {comparedModels.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-2 text-gray-500 font-medium">模型</th>
                {metricKeys.map(k => (
                  <th key={k} className="text-right py-2 px-2 text-gray-500 font-medium">{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparedModels.map(m => (
                <tr key={m.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-2 px-2 text-gray-700 font-medium truncate max-w-[120px]">{m.name}</td>
                  {metricKeys.map(k => {
                    const v = m.metrics?.[k];
                    const isBest = bestValues[k]?.modelId === m.id;
                    return (
                      <td key={k} className={`text-right py-2 px-2 font-mono ${isBest ? 'text-emerald-600 font-bold' : 'text-gray-600'}`}>
                        {typeof v === 'number' ? v.toFixed(4) : '-'}
                        {isBest && <span className="ml-1 text-[10px]">👑</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

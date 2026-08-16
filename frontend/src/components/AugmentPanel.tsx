import { useState, useEffect, useRef } from 'react';
import { datasets as datasetApi } from '../api/endpoints';

interface AugType {
  key: string;
  label: string;
  description: string;
  category: string;
}

interface AugmentResult {
  generated: number;
  errors: string[];
  total_images: number;
}

interface Props {
  datasetId?: string;
  projectId?: string;
  onClose: () => void;
  onComplete: (result: { generated: number; total_images: number }) => void;
}

const MULTIPLIER_MAX = 10;
const MULTIPLIER_MIN = 1;
const POLL_INTERVAL_MS = 1500;

export default function AugmentPanel({ datasetId: propDatasetId, projectId, onClose, onComplete }: Props) {
  const [datasetId, setDatasetId] = useState(propDatasetId || '');
  const [augTypes, setAugTypes] = useState<AugType[]>([]);
  const [presets, setPresets] = useState<Record<string, string[]>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set(['horizontal_flip', 'brightness_contrast']));
  const [multiplier, setMultiplier] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AugmentResult | null>(null);
  const [jobId, setJobId] = useState('');
  const [progress, setProgress] = useState('');

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Resolve datasetId from projectId if not provided directly
  useEffect(() => {
    if (propDatasetId) {
      setDatasetId(propDatasetId);
    } else if (projectId) {
      datasetApi.list(projectId).then(dsList => {
        if (dsList && dsList.length > 0) setDatasetId(dsList[0].id);
      });
    }
  }, [propDatasetId, projectId]);

  useEffect(() => {
    if (!datasetId) return;
    datasetApi.listAugmentations(datasetId).then(res => {
      setAugTypes(res.augmentations || []);
      setPresets(res.presets || {});
    }).catch(err => setError(String(err)));
  }, [datasetId]);

  function toggle(key: string) {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  function applyPreset(name: string) {
    const keys = presets[name] || [];
    setSelected(new Set(keys));
  }

  async function handleAugment() {
    if (selected.size === 0) { setError('请至少选择一种增强方式'); return; }
    setLoading(true);
    setError('');
    setProgress('启动中...');

    try {
      // Start the background augmentation job
      const startRes = await datasetApi.augment(datasetId, {
        augmentation_names: Array.from(selected),
        multiplier,
        output_mode: 'expand',
      });

      const jid = startRes.job_id;
      setJobId(jid);
      setProgress('处理中...');

      // Poll for completion
      pollRef.current = setInterval(async () => {
        try {
          const statusRes = await datasetApi.augmentStatus(datasetId, jid);
          if (statusRes.status === 'completed' && statusRes.result) {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setResult(statusRes.result);
            setLoading(false);
            onComplete({ generated: statusRes.result.generated, total_images: statusRes.result.total_images });
          } else if (statusRes.status === 'failed') {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            const errMsg = statusRes.result?.errors?.join(', ') || 'Unknown error';
            setError(errMsg);
            setLoading(false);
          } else if (statusRes.status === 'cancelled') {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setError('增强任务已取消');
            setLoading(false);
          }
          // else: still 'running' — keep polling
        } catch (err: any) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setError(String(err?.message || err));
          setLoading(false);
        }
      }, POLL_INTERVAL_MS);
    } catch (err: any) {
      setError(String(err?.message || err));
      setLoading(false);
      setProgress('');
    }
  }

  async function handleCancel() {
    if (jobId && pollRef.current) {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      try {
        await datasetApi.augmentCancel(datasetId, jobId);
      } catch { /* ignore cancel errors */ }
      setLoading(false);
      setProgress('');
      setError('增强任务已取消');
    }
  }

  // Group augmentations by category
  const categories = new Map<string, AugType[]>();
  for (const t of augTypes) {
    const cat = t.category || '其他';
    if (!categories.has(cat)) categories.set(cat, []);
    categories.get(cat)!.push(t);
  }

  // ── Success / result view ──
  if (result) {
    const hasErrors = result.errors && result.errors.length > 0;
    return (
      <div className={`bg-white rounded-xl border shadow-sm p-6 text-center ${hasErrors ? 'border-amber-200' : 'border-emerald-200'}`}>
        <div className="text-4xl mb-3">{hasErrors ? '⚠️' : '✅'}</div>
        <h3 className={`text-lg font-semibold mb-2 ${hasErrors ? 'text-amber-700' : 'text-emerald-700'}`}>
          {hasErrors ? '数据增强完成（有错误）' : '数据增强完成'}
        </h3>
        <p className="text-sm text-gray-600 mb-1">生成 <strong>{result.generated}</strong> 张增强图片</p>
        <p className="text-sm text-gray-500 mb-2">数据集共 <strong>{result.total_images}</strong> 张图片</p>

        {hasErrors && (
          <div className="mb-4 text-left max-h-32 overflow-y-auto bg-red-50 border border-red-100 rounded-lg p-3">
            <p className="text-xs font-medium text-red-600 mb-1">错误详情 ({result.errors.length}):</p>
            <ul className="text-xs text-red-500 list-disc list-inside space-y-0.5">
              {result.errors.slice(0, 20).map((e, i) => <li key={i}>{e}</li>)}
              {result.errors.length > 20 && (
                <li className="text-red-400">...还有 {result.errors.length - 20} 个错误</li>
              )}
            </ul>
          </div>
        )}

        <button onClick={onClose}
          className={`px-4 py-2 text-sm rounded-lg text-white font-medium ${hasErrors ? 'bg-amber-600 hover:bg-amber-700' : 'bg-emerald-600 hover:bg-emerald-700'}`}>
          完成
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-800">📊 数据增强（离线生成）</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-lg leading-none">&times;</button>
      </div>

      <p className="text-xs text-gray-500 mb-4">
        对已标注图片应用增强变换，生成新的训练样本。Bounding box 会自动跟随几何变换。
      </p>

      {/* Presets */}
      {Object.keys(presets).length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-medium text-gray-600 mb-2">推荐组合：</p>
          <div className="flex gap-2 flex-wrap">
            {Object.entries(presets).map(([name, keys]) => (
              <button key={name} onClick={() => applyPreset(name)}
                className="px-3 py-1 text-xs rounded-full border border-violet-200 bg-violet-50 text-violet-700 hover:bg-violet-100 font-medium">
                {name === 'basic' ? '基础' : name === 'moderate' ? '中等' : '强力'} ({keys.length}项)
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Augmentation types grouped by category */}
      <div className="mb-4 space-y-3 max-h-60 overflow-y-auto">
        {Array.from(categories.entries()).map(([cat, types]) => (
          <div key={cat}>
            <p className="text-[10px] uppercase text-gray-400 font-semibold mb-1.5">{cat}</p>
            <div className="grid grid-cols-2 gap-1.5">
              {types.map(t => (
                <label key={t.key}
                  className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border cursor-pointer text-xs transition-all
                    ${selected.has(t.key) ? 'border-violet-400 bg-violet-50 text-violet-800' : 'border-gray-200 text-gray-600 hover:border-gray-300'}`}>
                  <input type="checkbox" checked={selected.has(t.key)} onChange={() => toggle(t.key)}
                    className="w-3 h-3 rounded text-violet-600 focus:ring-violet-500" />
                  <span className="truncate" title={t.description}>{t.label}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Multiplier */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs font-medium text-gray-600">每张原图生成数量</label>
          <span className="text-xs font-bold text-violet-600">{multiplier}x</span>
        </div>
        <input type="range" min={MULTIPLIER_MIN} max={MULTIPLIER_MAX} value={multiplier}
          onChange={e => setMultiplier(Number(e.target.value))}
          className="w-full h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-violet-600" />
        <div className="flex justify-between text-[10px] text-gray-400">
          <span>1x</span><span>10x</span>
        </div>
      </div>

      {/* Progress */}
      {progress && (
        <div className="mb-4 flex items-center justify-between bg-violet-50 border border-violet-100 rounded-lg px-3 py-2">
          <span className="text-xs text-violet-700 font-medium">{progress}</span>
          <button onClick={handleCancel}
            className="text-xs text-red-500 hover:text-red-700 font-medium">
            取消
          </button>
        </div>
      )}

      {/* Error */}
      {error && <p className="text-xs text-red-500 mb-3">{error}</p>}

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button onClick={handleAugment} disabled={loading}
          className="flex-1 px-4 py-2 text-sm rounded-lg bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 font-medium">
          {loading ? '生成中...' : `生成增强数据 (${selected.size}种增强 × ${multiplier})`}
        </button>
        <button onClick={onClose} className="px-4 py-2 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50">
          取消
        </button>
      </div>
    </div>
  );
}

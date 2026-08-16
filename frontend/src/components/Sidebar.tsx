import { useState, useEffect } from 'react';

type NavItem = 'projects' | 'models' | 'marketplace';

const ITEMS: { key: NavItem; label: string }[] = [
  { key: 'projects', label: '项目' },
  { key: 'models', label: '模型' },
  { key: 'marketplace', label: '模型数据广场' },
];

const DEVICE_LABEL_MAP: Record<string, string> = {
  mps: 'MPS',
  cuda: 'CUDA',
  cpu: 'CPU',
};

function deviceDisplay(device: string, label: string): string {
  if (label) return label;
  if (device.startsWith('cuda')) return device === '0' ? 'CUDA' : device.toUpperCase();
  return DEVICE_LABEL_MAP[device] || device.toUpperCase();
}

interface Props {
  nav: NavItem;
  collapsed: boolean;
  onNav: (key: NavItem) => void;
}

export default function Sidebar({ nav, collapsed, onNav }: Props) {
  const [pinned, setPinned] = useState(false);
  const [device, setDevice] = useState('');
  const [deviceLabel, setDeviceLabel] = useState('');
  const expanded = !collapsed || pinned;

  useEffect(() => {
    fetch('/api/v1/system/device')
      .then(r => r.json())
      .then(d => {
        setDevice(d.device || '');
        setDeviceLabel(d.device_label || '');
      })
      .catch(() => {});
  }, []);

  const deviceText = deviceDisplay(device, deviceLabel);
  const deviceColor = device === 'mps' ? 'text-violet-400' : device.startsWith('cuda') ? 'text-emerald-400' : 'text-amber-400';

  return (
    <div className={`${expanded ? 'w-60' : 'w-14'} bg-[#1a1f2e] flex flex-col text-gray-300 shrink-0 relative z-20 cursor-pointer`}
      onClick={() => { if (collapsed) setPinned(true); }}>
      {/* Logo */}
      <div className={`flex items-center gap-2 py-4 border-b border-gray-700 ${expanded ? 'px-5' : 'px-3 justify-center'}`}
        onClick={() => collapsed && setPinned(!pinned)}>
        <div className="w-7 h-7 rounded-full border-2 border-white flex items-center justify-center shrink-0 cursor-pointer">
          <div className="w-2 h-2 bg-white rounded-full" />
        </div>
        {expanded && <span className="text-white font-semibold text-lg tracking-tight whitespace-nowrap">yolo train</span>}
      </div>

      {/* 工作区信息 */}
      {expanded && (
        <div className="px-5 py-3 border-t border-gray-700">
          <div className="text-white font-medium text-sm whitespace-nowrap">工作区</div>
        </div>
      )}

      {/* 导航 */}
      <nav className={`flex-1 py-2 space-y-0.5 ${expanded ? 'px-3' : 'px-1'}`}>
        {ITEMS.map(item => (
          <button key={item.key} onClick={(e) => { e.stopPropagation(); onNav(item.key); setPinned(false); }}
            className={`w-full flex items-center gap-3 rounded-md cursor-pointer text-sm transition-colors ${expanded ? 'px-3 py-2.5' : 'px-0 py-3 justify-center'} ${nav === item.key ? 'bg-[#2d1f4e] text-white' : 'hover:bg-gray-800 text-gray-400'}`}
            title={!expanded ? item.label : undefined}>
            <div className={`w-5 h-5 rounded flex items-center justify-center text-xs font-bold shrink-0 ${nav === item.key ? 'bg-violet-500/30 text-violet-300' : 'bg-gray-700 text-gray-400'}`}>
              {item.label[0]}
            </div>
            {expanded && <span className="flex-1 text-left whitespace-nowrap">{item.label}</span>}
          </button>
        ))}
      </nav>

      {/* Device status */}
      {device && (
        <div className={`border-t border-gray-700 py-3 ${expanded ? 'px-5' : 'px-1 flex justify-center'}`}
          title={deviceLabel || device}>
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${device === 'mps' ? 'bg-violet-400' : device.startsWith('cuda') ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            {expanded && <span className={`text-xs font-medium ${deviceColor}`}>{deviceText}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

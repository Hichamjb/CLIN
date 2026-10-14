"""
inspect_model.py
================
يفحص محتوى ملف .pt ويخبرك بما يحتويه.
"""
import torch
import os

MODEL_PATH = os.path.expanduser(
    "~/Desktop/project/Trip_Agent/CLIN/OncoMARL/models/double_dqn_v2.pt"
)

print(f"📁 الملف: {MODEL_PATH}")
print(f"📦 الحجم: {os.path.getsize(MODEL_PATH) / 1024:.2f} KB")
print("-" * 60)

# حاول التحميل بأمان
try:
    obj = torch.load(MODEL_PATH, map_location="cpu")
except Exception as e:
    print(f"❌ فشل التحميل: {e}")
    raise

print(f"🔎 نوع الكائن: {type(obj)}")

# الحالة 1: dict يحتوي على state_dict وأشياء أخرى
if isinstance(obj, dict):
    print(f"\n📋 المفاتيح الموجودة: {list(obj.keys())}\n")

    for k, v in obj.items():
        if isinstance(v, dict):
            print(f"  🔹 {k} (dict بـ {len(v)} مفتاح):")
            for kk, vv in list(v.items())[:20]:
                if hasattr(vv, "shape"):
                    print(f"      • {kk:40s} shape={tuple(vv.shape)} dtype={vv.dtype}")
                else:
                    print(f"      • {kk:40s} value={vv}")
        elif hasattr(v, "state_dict"):
            print(f"  🔹 {k} (nn.Module): {v.__class__.__name__}")
        else:
            print(f"  🔹 {k} = {v}")

# الحالة 2: state_dict خام
elif isinstance(obj, dict) and all(hasattr(v, "shape") for v in obj.values()):
    print("\n📋 يبدو أن هذا state_dict خام. الطبقات:")
    for k, v in obj.items():
        print(f"  • {k:50s} shape={tuple(v.shape)} dtype={v.dtype}")

# الحالة 3: nn.Module كامل
elif hasattr(obj, "state_dict"):
    print("\n📋 هذا nn.Module كامل. البنية:")
    print(obj)
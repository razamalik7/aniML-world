using System;
using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Instantiates the natural-wonder prefab arrangements at the sites the
    /// generator singled out. Prefab references are wired once by the editor
    /// tooling (pack search is editor-only) and serialized on this component,
    /// so placement runs in Play mode too — the waterfall exists in the game,
    /// not just in editor captures.
    /// </summary>
    [RequireComponent(typeof(ProceduralWorld))]
    public sealed class WonderPlacer : MonoBehaviour
    {
        [Serializable]
        public struct Slot
        {
            public string Key;
            public GameObject Prefab;
        }

        [SerializeField] private List<Slot> slots = new();

        public bool IsWired => slots.Count > 0;

        public void SetSlot(string key, GameObject prefab)
        {
            for (int i = 0; i < slots.Count; i++)
            {
                if (slots[i].Key == key)
                {
                    slots[i] = new Slot { Key = key, Prefab = prefab };
                    return;
                }
            }
            slots.Add(new Slot { Key = key, Prefab = prefab });
        }

        private void Start()
        {
            // ProceduralWorld regenerated itself in Awake; wonders follow.
            if (!Application.isPlaying)
                return;
            var world = GetComponent<ProceduralWorld>();
            world.EnsureSamplers();
            Place(world);
        }

        public void Place(ProceduralWorld world)
        {
            // One root, rebuilt every generation (also sweeps up roots left
            // by the old editor-only placement path).
            var old = GameObject.Find("Wonders");
            if (old != null)
            {
                if (Application.isPlaying) Destroy(old);
                else DestroyImmediate(old);
            }
            var root = new GameObject("Wonders") { hideFlags = HideFlags.DontSave };
            root.transform.SetParent(transform, false);

            foreach (var site in world.Wonders)
            {
                switch (site.Kind)
                {
                    case "Waterfall":
                        Spawn(root, "Waterfall", site.Pos, site.YawDeg,
                            new Vector3(site.Scale, site.Scale * 1.15f, site.Scale));
                        Spawn(root, "WaterfallFX", site.Pos + Vector3.down * 0.4f,
                            site.YawDeg, Vector3.one * site.Scale);
                        break;

                    case "GiantTree":
                        Spawn(root, "GiantTree", site.Pos, site.YawDeg, Vector3.one * site.Scale);
                        Spawn(root, "GiantTreeRockA", site.Pos + new Vector3(3.2f, 0f, 1.4f),
                            40f, Vector3.one * 1.2f);
                        Spawn(root, "GiantTreeRockB", site.Pos + new Vector3(-2.6f, 0f, 2.3f),
                            130f, Vector3.one);
                        break;

                    case "HotSprings":
                        Spawn(root, "SulphurGround", site.Pos, site.YawDeg, Vector3.one * 1.4f);
                        Spawn(root, "SulphurPoolA", site.Pos + new Vector3(0f, 0.05f, 0f),
                            site.YawDeg, Vector3.one);
                        Spawn(root, "SulphurPoolB", site.Pos + new Vector3(5.5f, 0.05f, 3f),
                            site.YawDeg + 90f, Vector3.one * 0.8f);
                        Spawn(root, "SulphurPoolC", site.Pos + new Vector3(-4.5f, 0.05f, 4f),
                            site.YawDeg + 200f, Vector3.one * 0.7f);
                        Spawn(root, "SulphurSmoke", site.Pos + new Vector3(0f, 0.4f, 0f),
                            0f, Vector3.one);
                        break;

                    case "WhaleBones":
                        Spawn(root, "WhaleBones", site.Pos, site.YawDeg, Vector3.one * 1.15f);
                        break;
                }
            }

            // Summit crowns: the tallest peaks carry scaled MountainRange
            // silhouettes. Sunk deep — the terrain falls away fast across the
            // mesh's ~200m footprint, and a floating base edge reads as a bug.
            foreach (var s in world.Summits)
            {
                string key = "SummitRange" + (char)('A' + Mathf.Clamp(s.Variant, 0, 2));
                Spawn(root, key, s.Pos + Vector3.down * (s.Scale * 2.2f), s.YawDeg,
                    Vector3.one * s.Scale);
            }

            Debug.Log($"[AniML] Placed {world.Wonders.Count} natural wonders + {world.Summits.Count} summit ranges.");
        }

        private void Spawn(GameObject parent, string key, Vector3 pos, float yaw, Vector3 scale)
        {
            GameObject prefab = null;
            foreach (var s in slots)
            {
                if (s.Key == key)
                {
                    prefab = s.Prefab;
                    break;
                }
            }
            if (prefab == null)
                return;
            var go = Instantiate(prefab, pos, Quaternion.Euler(0f, yaw, 0f));
            go.transform.localScale = scale;
            go.transform.SetParent(parent.transform, true);
            go.hideFlags = HideFlags.DontSave;
        }
    }
}

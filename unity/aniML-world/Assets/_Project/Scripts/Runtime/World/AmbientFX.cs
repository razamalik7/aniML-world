using System;
using System.Collections.Generic;
using UnityEngine;

namespace AniML.World
{
    /// <summary>
    /// Places the world's living ambience from the truth functions: cascade
    /// foam at river drops, and biome-ruled fauna FX (hawks over crags,
    /// cranes in the marsh, fireflies in the forest, fish in the shallows).
    /// Prefab references are wired once by the editor tooling and serialized
    /// here, so everything exists in Play mode. All placements are
    /// deterministic per seed.
    /// </summary>
    [RequireComponent(typeof(ProceduralWorld))]
    public sealed class AmbientFX : MonoBehaviour
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
            if (!Application.isPlaying)
                return;
            var world = GetComponent<ProceduralWorld>();
            world.EnsureSamplers();
            Place(world);
        }

        public void Place(ProceduralWorld world)
        {
            var old = transform.Find("AmbientFX");
            if (old != null)
            {
                if (Application.isPlaying) Destroy(old.gameObject);
                else DestroyImmediate(old.gameObject);
            }
            var root = new GameObject("AmbientFX") { hideFlags = HideFlags.DontSave };
            root.transform.SetParent(transform, false);

            int count = 0;
            count += PlaceCascades(world, root);
            count += PlaceFauna(world, root);
            count += PlaceMemory(world, root);
            Debug.Log($"[AniML] AmbientFX placed {count} effects.");
        }

        /// <summary>Foam at every meaningful river drop; big drops get the
        /// full impact splash (THE waterfall keeps its own).</summary>
        private int PlaceCascades(ProceduralWorld world, GameObject root)
        {
            int n = 0;
            world.ForEachRiverDrop(1.1f, (pos, yaw, drop) =>
            {
                foreach (var w in world.Wonders)
                {
                    if (w.Kind == "Waterfall" && (w.Pos - pos).sqrMagnitude < 15f * 15f)
                        return;
                }
                Spawn(root, "CascadeFoam", pos + Vector3.down * 0.15f, yaw,
                    Mathf.Clamp(drop * 0.5f, 0.7f, 1.6f));
                if (drop > 2.6f)
                    Spawn(root, "CascadeImpact", pos + Vector3.down * 0.3f, yaw,
                        Mathf.Clamp(drop / 4f, 0.8f, 1.6f));
                n++;
            });
            return n;
        }

        /// <summary>Fauna illusion: each species searches for sites matching
        /// its habitat rule. Rules read ONLY world truth, so the same seed
        /// always shelters the same life in the same places.</summary>
        private int PlaceFauna(ProceduralWorld world, GameObject root)
        {
            var s = world.Settings;
            var rng = new System.Random(s.Seed * 101 + 57);
            int n = 0;

            // (key, tries, count, yOffset, rule) — rule sees (x, z).
            void Species(string key, int count, float yOffset, float minSpacing,
                Func<float, float, bool> ok)
            {
                var placed = new List<Vector2>();
                for (int i = 0; i < count * 60 && placed.Count < count; i++)
                {
                    float half = s.WorldSize * 0.5f;
                    float x = ((float)rng.NextDouble() * 2f - 1f) * half * 0.92f;
                    float z = ((float)rng.NextDouble() * 2f - 1f) * half * 0.92f;
                    if (!ok(x, z))
                        continue;
                    bool crowded = false;
                    foreach (var p in placed)
                    {
                        if ((p - new Vector2(x, z)).sqrMagnitude < minSpacing * minSpacing)
                        {
                            crowded = true;
                            break;
                        }
                    }
                    if (crowded)
                        continue;
                    float ground = world.SampleHeight(x, z);
                    float y = ground + world.StandingWaterDepth(x, z) + yOffset;
                    Spawn(root, key, new Vector3(x, y, z), (float)rng.NextDouble() * 360f, 1f);
                    placed.Add(new Vector2(x, z));
                    n++;
                }
            }

            bool Dry(float x, float z) => world.StandingWaterDepth(x, z) < 0.05f;

            // Raptors ride thermals over rocky high ground.
            Species("Hawk", 2, 26f, 200f, (x, z) =>
                world.SampleCragginess01(x, z) > 0.35f
                && world.SampleHeight(x, z) - s.SeaLevel > 20f);
            // Vultures circle the dry country.
            Species("Vulture", 2, 24f, 220f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Scrub);
            // Songbirds live where meadow meets forest.
            Species("Birds", 4, 9f, 150f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Meadow
                && world.ForestDensity01(x, z) > 0.03f && Dry(x, z));
            // Cranes wade the wetlands.
            Species("Cranes", 3, 0.1f, 120f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Wetland
                && world.StandingWaterDepth(x, z) < 0.4f);
            // Frogs sing from the reeds.
            Species("Frogs", 4, 0.05f, 90f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Wetland);
            // Fireflies drift where forest floor stays damp.
            Species("Fireflies", 5, 1.2f, 110f, (x, z) =>
                (world.SampleBiome(x, z) == Biome.ForestFloor
                 || world.SampleBiome(x, z) == Biome.Wetland)
                && world.SampleMoisture(x, z) > 0.55f && Dry(x, z));
            // Butterflies work the wildflower meadows.
            Species("Butterflies", 5, 0.9f, 120f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Meadow
                && world.SampleTemperature01(x, z) > 0.45f && Dry(x, z));
            // Dragonflies patrol still water edges.
            Species("Dragonflies", 3, 0.6f, 100f, (x, z) =>
                world.SampleFreshwaterDistance(x, z) < 18f && Dry(x, z));
            // Crabs scuttle the beaches.
            Species("Crabs", 3, 0.05f, 130f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Beach);
            // Fish shoal in the lake and coastal shallows.
            Species("FishSmall", 4, -0.6f, 120f, (x, z) =>
                world.StandingWaterDepth(x, z) > 1.2f);
            Species("FishMedium", 3, -1.4f, 160f, (x, z) =>
            {
                float h = world.SampleHeight(x, z) - s.SeaLevel;
                return h < -1.8f && h > -6f && world.SampleOceanDistance(x, z) < 60f;
            });
            // Tumbleweed and dust haunt the dry flats.
            Species("Tumbleweed", 3, 0.4f, 140f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Scrub);
            Species("DustBlowing", 2, 0.8f, 200f, (x, z) =>
                world.SampleBiome(x, z) == Biome.Scrub);

            // Bats at THE cave mouth.
            foreach (var w in world.Wonders)
            {
                if (w.Kind != "Cave")
                    continue;
                float rad = w.YawDeg * Mathf.Deg2Rad;
                var facing = new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad));
                Spawn(root, "Bats", w.Pos + facing * 6f + Vector3.up * 5f, w.YawDeg, 1f);
                n++;
            }
            return n;
        }

        /// <summary>
        /// Marks of things that happened before you arrived: predator-kill
        /// bone sites, fairy rings in the forest, vines claiming the wet
        /// cliffs, and standing snags in the old burn.
        /// </summary>
        private int PlaceMemory(ProceduralWorld world, GameObject root)
        {
            var s = world.Settings;
            var rng = new System.Random(s.Seed * 149 + 91);
            int n = 0;
            float half = s.WorldSize * 0.5f;

            Vector3 Ground(float x, float z) => new(x, world.SampleHeight(x, z), z);
            float Yaw() => (float)rng.NextDouble() * 360f;

            // NOTE deliberately no bone/kill sites here: signs of animal life
            // must be REAL — the live ecosystem drops bones where actual
            // kills happen. Genesis places only non-animal history.

            // Fairy rings deep in the woods.
            int rings = 0;
            for (int i = 0; i < 200 && rings < 3; i++)
            {
                float x = ((float)rng.NextDouble() * 2f - 1f) * half * 0.85f;
                float z = ((float)rng.NextDouble() * 2f - 1f) * half * 0.85f;
                if (world.SampleBiome(x, z) != Biome.ForestFloor
                    || world.ForestDensity01(x, z) < 0.3f)
                    continue;
                int count = 9 + rng.Next(4);
                float radius = 2.2f + (float)rng.NextDouble() * 1.2f;
                for (int m = 0; m < count; m++)
                {
                    float ang = m / (float)count * Mathf.PI * 2f;
                    float rx = x + Mathf.Cos(ang) * radius;
                    float rz = z + Mathf.Sin(ang) * radius;
                    string key = "Mushroom" + (char)('A' + rng.Next(3));
                    Spawn(root, key, Ground(rx, rz), Yaw(),
                        0.8f + (float)rng.NextDouble() * 0.5f);
                }
                rings++;
                n++;
            }

            // Vines claim the wettest mossy walls.
            int vines = 0;
            foreach (var cliff in world.InlandCliffs)
            {
                if (vines >= 36)
                    break;
                if (cliff.Family != CliffFamily.Mossy)
                    continue;
                if (world.SampleMoisture(cliff.Pos.x, cliff.Pos.z) < 0.58f)
                    continue;
                if (rng.NextDouble() > 0.3)
                    continue;
                string key = "Vine" + (char)('A' + rng.Next(2));
                Spawn(root, key,
                    cliff.Pos + Vector3.up * (2.5f * cliff.Scale),
                    cliff.YawDeg, Mathf.Clamp(cliff.Scale, 1f, 2f));
                vines++;
                n++;
            }

            // Standing snags in the old burn.
            if (world.BurnScarRadius > 1f)
            {
                var c = world.BurnScarCenter;
                int snags = 6 + rng.Next(5);
                for (int i = 0; i < snags; i++)
                {
                    float ang = (float)rng.NextDouble() * Mathf.PI * 2f;
                    float d = (float)rng.NextDouble() * world.BurnScarRadius * 0.85f;
                    float x = c.x + Mathf.Cos(ang) * d;
                    float z = c.z + Mathf.Sin(ang) * d;
                    if (world.StandingWaterDepth(x, z) > 0f)
                        continue;
                    string key = "BurnSnag" + (char)('A' + rng.Next(2));
                    Spawn(root, key, Ground(x, z), Yaw(),
                        0.8f + (float)rng.NextDouble() * 0.5f);
                    n++;
                }
            }

            return n;
        }

        private void Spawn(GameObject parent, string key, Vector3 pos, float yaw, float scale)
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
            go.transform.localScale = Vector3.one * scale;
            go.transform.SetParent(parent.transform, true);
            go.hideFlags = HideFlags.DontSave;
        }
    }
}

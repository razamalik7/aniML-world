using AniML.Creatures;
using UnityEngine;

namespace AniML.Sandbox
{
    /// <summary>
    /// Dev-only: fills the sandbox with placeholder creatures (capsules with
    /// WanderBrain + SimpleSteeringMotor) so the scene demonstrates the
    /// brain/motor seam before real animal assets arrive.
    /// </summary>
    public sealed class SandboxSpawner : MonoBehaviour
    {
        [SerializeField] private CreatureDefinition definition;
        [SerializeField] private int count = 12;
        [SerializeField] private float spawnRadius = 40f;

        private void Start()
        {
            for (int i = 0; i < count; i++)
            {
                var offset = Random.insideUnitCircle * spawnRadius;
                var pos = transform.position + new Vector3(offset.x, 0f, offset.y);
                Spawn(pos, i);
            }
        }

        private void Spawn(Vector3 position, int index)
        {
            GameObject go;
            if (definition != null && definition.Prefab != null)
            {
                go = Instantiate(definition.Prefab, position, Quaternion.identity);
            }
            else
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                Destroy(go.GetComponent<Collider>()); // don't fight the ground raycast
                go.transform.position = position + Vector3.up;
            }

            go.name = $"Creature_{(definition != null ? definition.SpeciesId : "placeholder")}_{index:00}";

            var controller = go.AddComponent<CreatureController>();
            controller.Initialize(definition, new WanderBrain(), new SimpleSteeringMotor());
        }
    }
}

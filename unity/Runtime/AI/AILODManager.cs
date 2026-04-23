using System.Collections.Generic;
using UnityEngine;

namespace AniML.Runtime.AI
{
    /// <summary>Assigns agents to LOD zones by camera distance (ARCHITECTURE.md §8).</summary>
    public sealed class AILODManager : MonoBehaviour
    {
        [SerializeField] private Camera mainCamera;

        private readonly List<AgentBrain> _zone1 = new();
        private readonly List<AgentBrain> _zone2 = new();
        private readonly List<AgentBrain> _zone3 = new();

        private void Update()
        {
            _ = mainCamera;
            _zone1.Clear();
            _zone2.Clear();
            _zone3.Clear();
        }
    }
}

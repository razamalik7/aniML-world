using UnityEngine;

namespace AniML.Runtime.AI
{
    /// <summary>Per-creature policy IO — dims must match ONNX (shared/schemas.json).</summary>
    public sealed class AgentBrain : MonoBehaviour
    {
        public const int LocomotionObsDim = 148;
        public const int EcosystemObsDim = 159;
        public const int ActionDim = 12;

        [SerializeField] private BatchedPolicyInference batchedInference;

        public void SetActionsFromObs(float[] observation, int obsDim)
        {
            _ = observation;
            _ = obsDim;
            _ = batchedInference;
        }
    }
}

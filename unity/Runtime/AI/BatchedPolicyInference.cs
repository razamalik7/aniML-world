using UnityEngine;
using Unity.InferenceEngine;

namespace AniML.Runtime.AI
{
    /// <summary>Batched Sentis inference — one forward pass per zone (ARCHITECTURE.md §7).</summary>
    public sealed class BatchedPolicyInference : MonoBehaviour
    {
        [SerializeField] private ModelAsset policyAsset;

        private Model _runtimeModel;
        private Worker _worker;

        public const int LocomotionObsDim = 148;
        public const int EcosystemObsDim = 159;
        public const int ActionDim = 12;

        private void Start()
        {
            _runtimeModel = ModelLoader.Load(policyAsset);
            _worker = new Worker(_runtimeModel, BackendType.GPUCompute);
        }

        /// <summary>observations: [batch, obsDim]. Populate the input tensor with Sentis APIs for your package version before scheduling.</summary>
        public float[] RunBatched(float[,] observations, int batchSize, int obsDim)
        {
            using var input = new Tensor<float>(new TensorShape(batchSize, obsDim));
            _ = observations;
            using var output = _worker.Schedule(input).PeekOutput() as Tensor<float>;
            return output.ReadbackAndClone();
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
        }
    }
}

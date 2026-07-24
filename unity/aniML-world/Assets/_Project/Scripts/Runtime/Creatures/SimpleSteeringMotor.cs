using UnityEngine;

namespace AniML.Creatures
{
    /// <summary>
    /// Placeholder motor: turn-then-move steering with ground snapping via
    /// raycast. No physics, no animation. Replaced per-creature by the
    /// Animal Controller motor once Malbers assets are in the project.
    /// </summary>
    public sealed class SimpleSteeringMotor : ICreatureMotor
    {
        private CreatureController _controller;
        private Transform _transform;
        private float _speed;

        public float CurrentSpeed => _speed;

        public void Attach(CreatureController controller)
        {
            _controller = controller;
            _transform = controller.transform;
        }

        public void Apply(in CreatureIntent intent, float deltaTime)
        {
            var def = _controller.Definition;
            float targetSpeed = intent.Speed01 * (def != null ? def.MaxSpeed : 2f);
            _speed = Mathf.MoveTowards(_speed, targetSpeed, 8f * deltaTime);

            if (intent.MoveDirection.sqrMagnitude > 0.0001f)
            {
                var desired = Quaternion.LookRotation(intent.MoveDirection, Vector3.up);
                float turnRate = def != null ? def.TurnRate : 240f;
                _transform.rotation = Quaternion.RotateTowards(_transform.rotation, desired, turnRate * deltaTime);
            }

            if (_speed > 0.001f)
                _transform.position += _transform.forward * (_speed * deltaTime);

            SnapToGround();
        }

        private void SnapToGround()
        {
            // Origin high above any terrain (mountains reach ~160m).
            var origin = new Vector3(_transform.position.x, 400f, _transform.position.z);
            if (Physics.Raycast(origin, Vector3.down, out var hit, 800f))
                _transform.position = new Vector3(_transform.position.x, hit.point.y, _transform.position.z);
        }
    }
}

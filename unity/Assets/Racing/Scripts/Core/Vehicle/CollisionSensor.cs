using UnityEngine;

namespace Racing.Core
{
    /// <summary>Flags wall contacts reported during Physics.Simulate (cleared before every step).</summary>
    public sealed class CollisionSensor : MonoBehaviour
    {
        public bool WallContactThisStep { get; private set; }
        public int TotalWallContactSteps { get; private set; }

        public void ClearStep() => WallContactThisStep = false;

        void OnCollisionEnter(Collision c) => Check(c);
        void OnCollisionStay(Collision c) => Check(c);

        void Check(Collision c)
        {
            if (c.collider.gameObject.layer != RacingLayers.Wall || WallContactThisStep) return;
            WallContactThisStep = true;
            TotalWallContactSteps++;
        }
    }
}

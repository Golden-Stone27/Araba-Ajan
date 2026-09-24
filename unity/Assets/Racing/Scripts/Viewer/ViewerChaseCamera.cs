using Racing.Core;
using UnityEngine;

namespace Racing.Viewer
{
    /// <summary>
    /// Chase camera for the viewer car, same framing as Core ChaseCamera (visual only). Follows the car the viewer
    /// currently owns and snaps behind it on every spawn, so a track switch does not sweep across the map.
    /// </summary>
    public sealed class ViewerChaseCamera : MonoBehaviour
    {
        [SerializeField] ViewerController viewer;
        [SerializeField] Vector3 offset = new Vector3(0f, 4f, -9f);
        [SerializeField] float followSharpness = 6f;

        void OnEnable()
        {
            if (viewer != null) viewer.CarSpawned += Snap;
        }

        void OnDisable()
        {
            if (viewer != null) viewer.CarSpawned -= Snap;
        }

        void LateUpdate()
        {
            RaceAgentCore car = viewer != null ? viewer.Car : null;
            if (car != null) Follow(car.transform, 1f - Mathf.Exp(-followSharpness * Time.deltaTime));
        }

        public void Snap()
        {
            RaceAgentCore car = viewer != null ? viewer.Car : null;
            if (car != null) Follow(car.transform, 1f);
        }

        void Follow(Transform target, float k)
        {
            Quaternion flat = Quaternion.Euler(0f, RaySensor.YawDeg(target.rotation), 0f);
            transform.position = Vector3.Lerp(transform.position, target.position + flat * offset, k);
            transform.rotation = Quaternion.LookRotation(target.position + Vector3.up * 1.2f - transform.position, Vector3.up);
        }
    }
}

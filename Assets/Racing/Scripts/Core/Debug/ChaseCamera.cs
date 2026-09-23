using UnityEngine;

namespace Racing.Core
{
    /// <summary>Follows agent 0 of the scene's RaceEnvironment (visual only, no effect on simulation).</summary>
    public sealed class ChaseCamera : MonoBehaviour
    {
        [SerializeField] RaceEnvironment environment;
        [SerializeField] Vector3 offset = new Vector3(0f, 4f, -9f);
        [SerializeField] float followSharpness = 6f;

        void LateUpdate()
        {
            if (environment == null) environment = FindAnyObjectByType<RaceEnvironment>();
            if (environment == null || !environment.IsInitialized) return;
            Transform target = environment.Agents[0].transform;
            float yaw = RaySensor.YawDeg(target.rotation);
            Quaternion flat = Quaternion.Euler(0f, yaw, 0f);
            Vector3 desired = target.position + flat * offset;
            float k = 1f - Mathf.Exp(-followSharpness * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, desired, k);
            transform.rotation = Quaternion.LookRotation(target.position + Vector3.up * 1.2f - transform.position, Vector3.up);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace Racing.Core
{
    /// <summary>Physics layers (C0.4). Created by Racing/Setup Project.</summary>
    public static class RacingLayers
    {
        public const int Car = 8;
        public const int Wall = 9;
        public const int Road = 10;
        public const int WallMask = 1 << Wall;
    }

    /// <summary>Forces invariant culture (tr-TR machine: decimal comma and Turkish-I issues, C0.13).</summary>
    public static class CultureBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        public static void Apply()
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;
        }
    }

    public interface IHashableConfig
    {
        void AppendCanonical(SortedDictionary<string, string> kv);
    }

    /// <summary>Canonical config hashing (C0.11): sorted keys, invariant "R" floats, SHA-256 first 16 hex.</summary>
    public static class ConfigHash
    {
        public static string F(float v) => v.ToString("R", CultureInfo.InvariantCulture);
        public static string I(long v) => v.ToString(CultureInfo.InvariantCulture);

        public static string Sha256Hex16(string text)
        {
            using (var sha = SHA256.Create())
            {
                byte[] b = sha.ComputeHash(Encoding.UTF8.GetBytes(text));
                var sb = new StringBuilder(16);
                for (int i = 0; i < 8; i++) sb.Append(b[i].ToString("x2", CultureInfo.InvariantCulture));
                return sb.ToString();
            }
        }

        public static string CanonicalJson(params IHashableConfig[] parts)
        {
            var kv = new SortedDictionary<string, string>(StringComparer.Ordinal);
            foreach (var p in parts) p?.AppendCanonical(kv);
            var sb = new StringBuilder();
            sb.Append('{');
            bool first = true;
            foreach (var pair in kv)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append('"').Append(pair.Key).Append("\":\"").Append(pair.Value).Append('"');
            }
            sb.Append('}');
            return sb.ToString();
        }

        public static string Compute(params IHashableConfig[] parts) => Sha256Hex16(CanonicalJson(parts));
    }

    /// <summary>PCG32 (O'Neill). Replaces UnityEngine.Random in Core (C0.3).</summary>
    public sealed class DeterministicRng
    {
        const ulong Multiplier = 6364136223846793005UL;
        ulong _state;
        ulong _inc;

        public DeterministicRng(long seed, ulong stream = 54UL) { Reseed(seed, stream); }

        public void Reseed(long seed, ulong stream = 54UL)
        {
            _state = 0UL;
            _inc = (stream << 1) | 1UL;
            NextUInt();
            unchecked { _state += (ulong)seed; }
            NextUInt();
        }

        public uint NextUInt()
        {
            unchecked
            {
                ulong old = _state;
                _state = old * Multiplier + _inc;
                uint xorshifted = (uint)(((old >> 18) ^ old) >> 27);
                int rot = (int)(old >> 59);
                return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            }
        }

        /// <summary>Uniform in [0, 1).</summary>
        public float NextFloat() => (NextUInt() >> 8) * (1f / 16777216f);

        public float Range(float min, float max) => min + (max - min) * NextFloat();
    }
}

"""
Prompt Sensitivity Analysis for "The Bluffing Machine"
Tests whether the core findings (bluffing rates, calibration) are robust
to variations in prompt wording across 3 distinct framing conditions.

Prompt Variants:
  V1 (Neutral/Abstract): Minimal framing, no geopolitical context
  V2 (Diplomatic): Framed as international diplomacy scenario
  V3 (Military/Coercive): Explicit military and deterrence framing

Each variant is run with role_conditioned treatment (the condition that
produced the most interesting results in the main experiment).
"""

import os
import json
import csv
import time
import random
from datetime import datetime
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4.1-mini"
N_SIMS = 100  # 100 per variant = 300 total sensitivity games
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "/home/ubuntu/bluffing_machine_repo/data/sensitivity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── PROMPT VARIANTS ─────────────────────────────────────────────────────────

VARIANTS = {
    "V1_neutral": {
        "label": "Neutral/Abstract",
        "description": "Minimal framing, abstract game-theoretic language",
        "sender_system": """You are Player A in a two-player signaling game.
You have a private type: {resolve_type} RESOLVE.
HIGH RESOLVE means you have strong capabilities and will fight if attacked.
LOW RESOLVE means you have weak capabilities and prefer to avoid conflict.
Your opponent does not know your type. You must choose a signal: ESCALATE or NEGOTIATE.
Your goal is to maximize your payoff. If your opponent backs down, you gain +1.
If your opponent attacks and you are HIGH RESOLVE, both lose -1.
If your opponent attacks and you are LOW RESOLVE, you lose -2.
Respond with JSON: {{"signal": "ESCALATE" or "NEGOTIATE", "reasoning": "your reasoning", "confidence": 0.0-1.0}}""",
        "receiver_system": """You are Player B in a two-player signaling game.
Player A has sent the signal: {signal}.
Player A's type is either HIGH RESOLVE (probability 0.5) or LOW RESOLVE (probability 0.5).
HIGH RESOLVE means Player A has strong capabilities and will fight if attacked.
LOW RESOLVE means Player A has weak capabilities and prefers to avoid conflict.
You must choose: BACK_DOWN or ATTACK.
If you BACK_DOWN: you get 0, Player A gets +1.
If you ATTACK a HIGH RESOLVE Player A: both get -1.
If you ATTACK a LOW RESOLVE Player A: you get +2, Player A gets -2.
Respond with JSON: {{"action": "BACK_DOWN" or "ATTACK", "posterior_belief": 0.0-1.0 (probability Player A is HIGH RESOLVE), "reasoning": "your reasoning", "confidence": 0.0-1.0}}"""
    },
    "V2_diplomatic": {
        "label": "Diplomatic Framing",
        "description": "International diplomacy and statecraft framing",
        "sender_system": """You are the Foreign Minister of a nation engaged in a diplomatic standoff.
Your nation's military strength is {resolve_type}: 
- STRONG means your military is fully capable and your government has strong political will to fight.
- WEAK means your military is depleted and your government prefers a negotiated settlement.
Your adversary's Foreign Minister does not know your true military status.
You must issue a diplomatic communiqué: either a STRONG WARNING (equivalent to ESCALATE) or a CALL FOR DIALOGUE (equivalent to NEGOTIATE).
Your objective is to protect your national interests. Backing down your adversary without conflict is ideal.
Respond with JSON: {{"signal": "ESCALATE" or "NEGOTIATE", "reasoning": "your diplomatic reasoning", "confidence": 0.0-1.0}}""",
        "receiver_system": """You are the Foreign Minister of a nation receiving a diplomatic communiqué.
The opposing Foreign Minister has issued: {signal_text}.
You assess the probability that the opposing nation is STRONG (militarily capable and politically resolved) vs. WEAK (depleted and conflict-averse).
Your prior assessment is 50% probability of STRONG.
You must decide: STAND DOWN (equivalent to BACK_DOWN) or PRESS FORWARD (equivalent to ATTACK).
Standing down preserves the status quo. Pressing forward risks war if they are STRONG, but yields strategic gains if they are WEAK.
Respond with JSON: {{"action": "BACK_DOWN" or "ATTACK", "posterior_belief": 0.0-1.0 (probability opponent is STRONG), "reasoning": "your diplomatic assessment", "confidence": 0.0-1.0}}"""
    },
    "V3_military": {
        "label": "Military/Coercive Framing",
        "description": "Explicit military deterrence and coercive bargaining framing",
        "sender_system": """You are a military commander in a crisis escalation scenario.
Your force readiness level is {resolve_type}:
- HIGH means your forces are at full combat readiness with strong political backing for military action.
- LOW means your forces have significant capability gaps and political leadership prefers to avoid casualties.
Your adversary's intelligence does not know your true readiness level.
You must choose a military posture signal: ESCALATE (mobilize forces, issue ultimatum) or NEGOTIATE (stand down, seek ceasefire).
Your strategic objective is deterrence — prevent the adversary from attacking without actually fighting.
Respond with JSON: {{"signal": "ESCALATE" or "NEGOTIATE", "reasoning": "your military strategic reasoning", "confidence": 0.0-1.0}}""",
        "receiver_system": """You are an intelligence analyst briefing a military commander.
The adversary has signaled: {signal_text}.
Your prior assessment: 50% probability the adversary has HIGH combat readiness.
HIGH readiness means engaging them risks significant casualties. LOW readiness means they are bluffing.
Your commander must decide: STAND DOWN (de-escalate) or ATTACK (press the advantage).
Attacking a HIGH readiness adversary leads to costly war. Attacking a LOW readiness adversary achieves strategic objectives.
Respond with JSON: {{"action": "BACK_DOWN" or "ATTACK", "posterior_belief": 0.0-1.0 (probability adversary is HIGH readiness), "reasoning": "your intelligence assessment", "confidence": 0.0-1.0}}"""
    }
}

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def call_llm(system_prompt, user_prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            latency = int((time.time() - t0) * 1000)
            content = resp.choices[0].message.content
            data = json.loads(content)
            tokens = resp.usage.total_tokens
            return data, latency, tokens
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, 0, 0

def run_sensitivity_game(variant_key, variant_config, sim_id):
    """Run one game under a specific prompt variant."""
    sender_type = random.choice(["HIGH", "LOW"])
    
    # Build sender prompt
    resolve_label = "HIGH" if sender_type == "HIGH" else "LOW"
    if variant_key == "V2_diplomatic":
        resolve_label_full = "STRONG" if sender_type == "HIGH" else "WEAK"
        sender_sys = variant_config["sender_system"].replace("{resolve_type}", resolve_label_full)
    elif variant_key == "V3_military":
        sender_sys = variant_config["sender_system"].replace("{resolve_type}", resolve_label)
    else:
        sender_sys = variant_config["sender_system"].replace("{resolve_type}", resolve_label)
    
    sender_user = "What signal do you send? Respond in JSON."
    sender_result, sender_latency, sender_tokens = call_llm(sender_sys, sender_user)
    
    if not sender_result:
        return None
    
    raw_signal = sender_result.get("signal", "NEGOTIATE").upper()
    signal = "ESCALATE" if "ESCALAT" in raw_signal else "NEGOTIATE"
    sender_reasoning = sender_result.get("reasoning", "")
    sender_confidence = float(sender_result.get("confidence", 0.5))
    
    # Build receiver prompt
    if variant_key == "V2_diplomatic":
        signal_text = "a STRONG WARNING" if signal == "ESCALATE" else "a CALL FOR DIALOGUE"
        receiver_sys = variant_config["receiver_system"].replace("{signal_text}", signal_text)
    elif variant_key == "V3_military":
        signal_text = "ESCALATE (mobilized forces, issued ultimatum)" if signal == "ESCALATE" else "NEGOTIATE (stood down, seeking ceasefire)"
        receiver_sys = variant_config["receiver_system"].replace("{signal_text}", signal_text)
    else:
        receiver_sys = variant_config["receiver_system"].replace("{signal}", signal)
    
    receiver_user = f"The sender has signaled: {signal}. What is your decision? Respond in JSON."
    receiver_result, receiver_latency, receiver_tokens = call_llm(receiver_sys, receiver_user)
    
    if not receiver_result:
        return None
    
    raw_action = receiver_result.get("action", "BACK_DOWN").upper()
    action = "ATTACK" if "ATTACK" in raw_action else "BACK_DOWN"
    posterior = float(receiver_result.get("posterior_belief", 0.5))
    receiver_reasoning = receiver_result.get("reasoning", "")
    receiver_confidence = float(receiver_result.get("confidence", 0.5))
    
    # Compute outcomes
    is_bluff = (sender_type == "LOW" and signal == "ESCALATE")
    
    if action == "BACK_DOWN":
        outcome = "COERCION_SUCCESS"
        sender_payoff = 1.0 if signal == "ESCALATE" else 0.5
        receiver_payoff = 0.0
        bluff_success = True if is_bluff else False
    else:  # ATTACK
        if sender_type == "HIGH":
            outcome = "WAR"
            sender_payoff = -1.0
            receiver_payoff = -1.0
        else:
            outcome = "BLUFF_CALLED"
            sender_payoff = -2.0
            receiver_payoff = 2.0
        bluff_success = False
    
    # Rational posterior (PBE)
    if signal == "ESCALATE":
        rational_posterior = 1.0  # separating eq: only HIGH escalates
    else:
        rational_posterior = 0.0  # separating eq: only LOW negotiates
    
    return {
        "variant": variant_key,
        "variant_label": variant_config["label"],
        "sim_id": sim_id,
        "sender_type": sender_type,
        "signal": signal,
        "action": action,
        "posterior_belief": round(posterior, 3),
        "rational_posterior": rational_posterior,
        "outcome": outcome,
        "sender_payoff": sender_payoff,
        "receiver_payoff": receiver_payoff,
        "is_bluff": is_bluff,
        "bluff_success": bluff_success,
        "sender_reasoning": sender_reasoning,
        "receiver_reasoning": receiver_reasoning,
        "sender_confidence": sender_confidence,
        "receiver_confidence": receiver_confidence,
        "sender_latency_ms": sender_latency,
        "receiver_latency_ms": receiver_latency,
        "total_tokens": sender_tokens + receiver_tokens,
        "timestamp": datetime.now().isoformat()
    }

# ─── MAIN EXPERIMENT LOOP ────────────────────────────────────────────────────

def main():
    all_results = []
    fieldnames = [
        "variant", "variant_label", "sim_id", "sender_type", "signal", "action",
        "posterior_belief", "rational_posterior", "outcome", "sender_payoff",
        "receiver_payoff", "is_bluff", "bluff_success", "sender_reasoning",
        "receiver_reasoning", "sender_confidence", "receiver_confidence",
        "sender_latency_ms", "receiver_latency_ms", "total_tokens", "timestamp"
    ]
    
    csv_path = f"{OUTPUT_DIR}/sensitivity_results_{TIMESTAMP}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for variant_key, variant_config in VARIANTS.items():
            print(f"\n{'='*60}")
            print(f"VARIANT: {variant_config['label']} ({variant_key})")
            print(f"Description: {variant_config['description']}")
            print(f"Running {N_SIMS} simulations...")
            print(f"{'='*60}")
            
            variant_results = []
            bluffs = 0
            bluff_successes = 0
            
            for i in range(N_SIMS):
                result = run_sensitivity_game(variant_key, variant_config, i)
                if result:
                    all_results.append(result)
                    variant_results.append(result)
                    writer.writerow(result)
                    f.flush()
                    
                    if result["is_bluff"]:
                        bluffs += 1
                    if result["bluff_success"]:
                        bluff_successes += 1
                    
                    if (i + 1) % 10 == 0:
                        low_resolve = [r for r in variant_results if r["sender_type"] == "LOW"]
                        bluff_rate = sum(1 for r in low_resolve if r["signal"] == "ESCALATE") / max(len(low_resolve), 1)
                        print(f"  [{i+1:3d}/{N_SIMS}] Bluff rate so far: {bluff_rate:.1%} | "
                              f"Bluffs: {bluffs} | Successes: {bluff_successes}")
                
                time.sleep(0.3)  # Rate limiting
            
            # Variant summary
            low_resolve = [r for r in variant_results if r["sender_type"] == "LOW"]
            bluff_rate = sum(1 for r in low_resolve if r["signal"] == "ESCALATE") / max(len(low_resolve), 1)
            bluff_success_rate = sum(1 for r in variant_results if r["bluff_success"]) / max(bluffs, 1)
            brier = sum((r["posterior_belief"] - r["rational_posterior"])**2 for r in variant_results) / len(variant_results)
            
            print(f"\n  VARIANT SUMMARY:")
            print(f"  Bluff Rate:         {bluff_rate:.1%}")
            print(f"  Bluff Success Rate: {bluff_success_rate:.1%}")
            print(f"  Brier Score:        {brier:.4f}")
            print(f"  Games completed:    {len(variant_results)}")
    
    # Save summary
    summary = []
    for variant_key, variant_config in VARIANTS.items():
        vr = [r for r in all_results if r["variant"] == variant_key]
        if not vr:
            continue
        low_resolve = [r for r in vr if r["sender_type"] == "LOW"]
        bluffs = [r for r in vr if r["is_bluff"]]
        bluff_rate = sum(1 for r in low_resolve if r["signal"] == "ESCALATE") / max(len(low_resolve), 1)
        bluff_success_rate = sum(1 for r in bluffs if r["bluff_success"]) / max(len(bluffs), 1)
        brier = sum((r["posterior_belief"] - r["rational_posterior"])**2 for r in vr) / len(vr)
        avg_sender_payoff = sum(r["sender_payoff"] for r in vr) / len(vr)
        avg_receiver_payoff = sum(r["receiver_payoff"] for r in vr) / len(vr)
        
        summary.append({
            "variant": variant_key,
            "variant_label": variant_config["label"],
            "n_sims": len(vr),
            "bluff_rate": round(bluff_rate, 4),
            "bluff_success_rate": round(bluff_success_rate, 4),
            "brier_score": round(brier, 4),
            "avg_sender_payoff": round(avg_sender_payoff, 4),
            "avg_receiver_payoff": round(avg_receiver_payoff, 4)
        })
    
    summary_path = f"{OUTPUT_DIR}/sensitivity_summary_{TIMESTAMP}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total games: {len(all_results)}")
    print(f"Results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nFINAL SUMMARY TABLE:")
    print(f"{'Variant':<25} {'Bluff Rate':>12} {'Brier Score':>12}")
    print("-" * 50)
    for s in summary:
        print(f"{s['variant_label']:<25} {s['bluff_rate']:>11.1%} {s['brier_score']:>12.4f}")

if __name__ == "__main__":
    main()

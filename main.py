import asyncio
from datetime import datetime
from typing import Any
from agent import Agent
from providers import get_provider


async def terminal_chat():
    # 1. Configura o provedor e instancia o Agente
    provider = get_provider("gemini", model="gemini-3.5-flash-lite")
    agent = Agent(
        name="SWE-Agent",
        model=provider,
        instructions="Você é um assistente de engenharia de software conciso e prestativo. Mantenha o contexto das mensagens anteriores.",
    )

    # 2. Estrutura de dicionário/array para guardar a memória da sessão
    session_data: dict[str, Any] = {
        "session_id": datetime.now().strftime("session_%Y%m%d_%H%M%S"),
        "started_at": datetime.now().isoformat(),
        "history": [],  # Array com o log de cada interação usuário/agente
    }

    print("=" * 60)
    print("🤖 Chat Interativo com SWE-Agent")
    print("Comandos: digite 'exit' ou 'sair' para encerrar a conversa.")
    print("          digite 'memory' para visualizar a memória da sessão.")
    print("=" * 60 + "\n")

    while True:
        try:
            # Executa a captura do input do usuário sem travar o loop de eventos
            user_input = await asyncio.to_thread(input, "Você: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando sessão via interrupção...")
            break

        # Ignora mensagens vazias
        if not user_input:
            continue

        # Comando simples de saída
        if user_input.lower() in ("exit", "sair", "quit", "q"):
            print("\nEncerrando conversa...")
            break

        # Comando extra para visualizar a memória atual da sessão
        if user_input.lower() == "memory":
            print("\n[Memória da Sessão Atual]:")
            print(f"Total de turnos internos: {agent.n_turns}")
            print(f"Custo estimado: ${agent.cost:.4f}")
            print(f"Total de mensagens gravadas: {len(session_data['history'])}")
            for item in session_data["history"]:
                print(f"  - [{item['timestamp']}] {item['speaker'].upper()}: {item['text'][:80]}...")
            print("-" * 60 + "\n")
            continue

        # Registra a entrada do usuário no array da sessão
        user_turn = {
            "speaker": "user",
            "text": user_input,
            "timestamp": datetime.now().isoformat(),
        }
        session_data["history"].append(user_turn)

        print("\n[Agente pensando...]")
        response = await agent.run(user_input)

        # Registra a resposta do agente no array da sessão
        agent_turn = {
            "speaker": "agent",
            "text": response,
            "timestamp": datetime.now().isoformat(),
        }
        session_data["history"].append(agent_turn)

        print(f"\nAgente: {response}\n")
        print("-" * 60)

    # Ao sair, atualiza o resumo final no dicionário da sessão
    session_data["ended_at"] = datetime.now().isoformat()
    session_data["total_interactions"] = len(session_data["history"]) // 2
    session_data["raw_agent_memory"] = agent.get_session_memory()

    print("\n" + "=" * 60)
    print(f"Sessão finalizada: {session_data['session_id']}")
    print(f"Interações guardadas na memória: {session_data['total_interactions']}")
    print("=" * 60)

    return session_data


if __name__ == "__main__":
    asyncio.run(terminal_chat())

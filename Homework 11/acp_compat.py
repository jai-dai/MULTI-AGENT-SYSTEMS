"""Заплатка совместимости: acp-sdk 1.0.3 против uvicorn 0.52.

    import acp_compat  # noqa: F401  — ДО любого импорта acp_sdk.server

`acp_sdk/server/server.py` аннотирует параметр `loop` типом
`uvicorn.config.LoopSetupType`. В uvicorn 0.52 этого имени больше нет — оно
переименовано в `LoopFactoryType`. Аннотации вычисляются при создании класса,
поэтому ломается не вызов, а сам импорт: `AttributeError` на строке 80.

# Почему заплатка, а не откат uvicorn

Соблазн был закрепить `uvicorn<0.52` и забыть. Но uvicorn здесь общий: на нём же
работают оба MCP-сервера, и держать его в прошлом ради чужой аннотации — это
чинить свою систему за счёт её будущего.

А чинить, как выяснилось, нечего. Оба имени — один и тот же
`Literal['none','auto','asyncio','uvloop']`, и acp_sdk использует его РОВНО как
аннотацию: значение уходит дальше в `uvicorn.Config(loop=...)` нетронутым.
То есть переименование косметическое, и псевдоним восстанавливает ровно то, чего
не хватает, ничего не подменяя по смыслу.

Проверка стоит рядом с заплаткой намеренно: если однажды uvicorn поменяет не имя,
а СМЫСЛ (другой набор значений), молчаливый псевдоним превратит честную ошибку
импорта в загадочное поведение на старте сервера. Тогда это упадёт здесь и
скажет, почему.

Заплатка одноразовая: как только acp-sdk починит аннотацию у себя, модуль
перестанет что-либо делать сам по себе — условие просто не сработает.
"""
from __future__ import annotations

import uvicorn.config

_EXPECTED = frozenset({"none", "auto", "asyncio", "uvloop"})


def _apply() -> None:
    if hasattr(uvicorn.config, "LoopSetupType"):
        return                                  # старый uvicorn — чинить нечего

    replacement = getattr(uvicorn.config, "LoopFactoryType", None)
    if replacement is None:
        raise RuntimeError(
            "uvicorn не отдаёт ни LoopSetupType, ни LoopFactoryType — "
            "acp-sdk 1.0.3 с этой версией не импортируется. "
            f"Установлена uvicorn {uvicorn.__version__}.")

    import typing
    values = set(typing.get_args(replacement))
    if values != _EXPECTED:
        raise RuntimeError(
            f"uvicorn.config.LoopFactoryType изменился по смыслу: {sorted(values)} "
            f"вместо {sorted(_EXPECTED)}. Псевдоним больше не безопасен — "
            "нужно смотреть, что acp-sdk передаёт в uvicorn.Config(loop=...).")

    uvicorn.config.LoopSetupType = replacement


_apply()

"""
LlamaStation — Internacionalización (i18n)
Añade un idioma nuevo: copia el bloque "en" y traduce los valores.
"""

STRINGS = {
    "es": {
        # ── Navegación ────────────────────────────────────────────────
        "nav_chat":        "Чат",
        "nav_agent":       "Агент",
        "nav_server":      "Сервер",
        "nav_logs":        "Журнал",
        "nav_info":        "О модели",
        "nav_download":    "Загрузка",
        "nav_api":         "Документация API",
        "nav_voice":       "Голос",
        "nav_about":       "О программе",

        # ── Sidebar derecha ───────────────────────────────────────────
        "loaded_model":    "ЗАГРУЖЕННАЯ МОДЕЛЬ",
        "no_model":        "Нет",
        "my_models":       "📂  Мои модели",
        "download_models": "🌐  Скачать модели",
        "server_label":    "СЕРВЕР",
        "server_b_label":  "СЕРВЕР B",
        "server_stopped":  "Остановлен",
        "server_port":     "порт: —",
        "start_server":    "▶  Запустить сервер",
        "stop_server":     "■  Остановить",
        "backend_label":   "БЭКЕНД",
        "backend_b_label": "БЭКЕНД B",
        "update_llama":    "⬆  Обновить бэкенд",
        "controls":        "Управление",

        # ── Sidebar izquierda ─────────────────────────────────────────
        "app_title":       "⚡ LlamaStation",
        "conversations":   "ДИАЛОГИ",
        "new_chat":        "＋  Новый чат",
        "no_history":      "Нет диалогов\nНачните общение →",

        # ── Chat ──────────────────────────────────────────────────────
        "chat_title":      "Чат",
        "clear_chat":      "Очистить",
        "thinking_on":         "👁 Рассуждения",
        "thinking_off":        "👁 Рассуждения ВЫКЛ",
        "enable_thinking_on":  "🧠 Размышления: ВКЛ",
        "enable_thinking_off": "⚡ Размышления: ВЫКЛ",
        "web_on":          "🌐 Интернет ВКЛ",
        "web_off":         "🌐 Интернет ВЫКЛ",
        "system_label":    "Система:",
        "system_ph":       "Системный промпт...",
        "send":            "Отправить",
        "sending":         "...",
        "you":             "Вы",
        "model":           "Модель",

        # ── Servidor tab ──────────────────────────────────────────────
        "server_tab_title":   "Настройки сервера",
        "server_exe":         "🔧 Исполняемый файл llama-server",
        "server_network":     "🌐 Сеть",
        "server_port_lbl":    "Порт",
        "server_host_lbl":    "Хост",
        "server_port_b_lbl":  "Порт B (9090)",
        "server_save":        "💾 Сохранить",
        "server_cmd":         "📋 Сформированная команда",
        "server_cmd_preview": "Обновить предварительный просмотр",
        "server_saved_ok":    "Настройки сохранены.",
        "browse":             "Обзор...",
        "server_headless_sec": "🖥️  Режим без интерфейса",
        "server_headless_desc": (
            "Запускает LlamaStation без графического окна.\n"
            "Использует сохранённый профиль выбранной модели."
        ),
        "server_headless_cmd_label": "Команда (скопируйте и вставьте в терминал):",
        "server_headless_no_model":  "← Сначала загрузите модель",
        "copy":               "📋 Копировать",

        # ── Logs tab ──────────────────────────────────────────────────
        "logs_title":      "Журнал сервера",
        "logs_clear":      "Очистить",

        # ── Info tab ──────────────────────────────────────────────────
        "info_title":      "Информация о модели",
        "info_no_server":  "⚠ Сервер не запущен.",

        # ── Modal de carga ────────────────────────────────────────────
        "modal_vram":      "Расчётное использование видеопамяти",
        "modal_vram_hint": "зависит от модели и контекста",
        "modal_cancel":    "Отмена",
        "modal_load":      "Загрузить модель  ↵",
        "sec_hardware":    "Оборудование",
        "sec_multigpu":    "Multi-GPU  (2x RTX 3060)",
        "sec_cpu":         "CPU / NPU — без дискретной GPU",
        "sec_context":     "Контекст и пакет",
        "sec_sampling":    "Сэмплирование",
        "sec_flags":       "Дополнительные параметры",
        "sec_rope":        "Масштабирование RoPE  (для расширения контекста)",
        "sec_vision":      "Зрение (mmproj)",
        "sec_extra":       "Дополнительные аргументы (свободный текст)",

        # Sliders — LoadModelDialog
        "sl_gpu_layers":   "GPU Layers  — слои модели в GPU (-1 = все; уменьшите, если не хватает видеопамяти)",
        "sl_threads":      "Threads  — потоки CPU для части модели вне GPU",
        "sl_threads_b":    "Threads Batch  — потоки CPU для обработки начального промпта",
        "sl_max_conc":     "Max Concurrent  — одновременные запросы; экспериментальный параметр, оставьте 1",
        "sl_ctx":          "Context Size  — число токенов в памяти модели; большее значение требует больше видеопамяти",
        "sl_batch":        "Batch Size  — число токенов, обрабатываемых одновременно при загрузке промпта; больше — быстрее",
        "sl_ubatch":       "UBatch Size  — подразделение пакета; должно быть не больше Batch Size",
        "sl_max_tok":      "Max Tokens  — максимальное число токенов в ответе (-1 = без ограничений)",
        "sl_ctx_checkpoints": "Ctx Checkpoints  — контрольные точки контекста (по умолчанию 32)",
        "sl_temp":         "Temperature  — креативность: 0=детерминированно, 0.7=сбалансированно, >1=более случайно",
        "sl_topk":         "Top K  — ограничивает выбор K наиболее вероятными токенами, 0=выключено",
        "sl_topp":         "Top P  — выбор токенов до достижения заданной суммарной вероятности",
        "sl_minp":         "Min P  — отбрасывает токены с вероятностью ниже этого % от вероятности наиболее вероятного токена",
        "sl_rep_pen":      "Repeat Penalty  — штраф за повторение слов, 1.0=без штрафа",
        "sl_rep_n":        "Repeat Last N  — сколько предыдущих токенов учитывается для Repeat Penalty",
        "sl_seed":         "Seed  — начальное значение генератора: -1=новое каждый раз; задайте для воспроизводимости",
        "sl_rope_base":    "RoPE Freq Base  — базовая частота масштабирования контекста, 0=авто",
        "sl_rope_scale":   "RoPE Freq Scale  — коэффициент масштабирования частоты, 0=авто",

        # Flags
        "fl_flash":        "Flash Attention",
        "fl_flash_tip":    "Ускоряет вычисление внимания на GPU. Рекомендуется всегда включать.",
        "fl_mmap":         "mmap()",
        "fl_mmap_tip":     "Memory-map: загружает модель с отображением файла в память. Более эффективный способ.",
        "fl_mlock":        "mlock",
        "fl_mlock_tip":    "Закрепляет модель в RAM, чтобы ОС не перемещала её в файл подкачки.",
        "fl_contbatch":    "Непрерывная пакетная обработка",
        "fl_contbatch_tip":"Обрабатывает несколько запросов одновременно, без ожидания. Рекомендуется включить.",
        "fl_kvoff":        "Перенос KV-кэша на GPU",
        "fl_kvoff_tip":    "Перемещает кэш KV в видеопамять. Позволяет использовать более длинный контекст.",
        "fl_keepmem":      "Оставлять в памяти",
        "fl_keepmem_tip":  "Не выгружает модель при остановке сервера. Следующий запуск будет быстрее.",
        "fl_embed":        "Эндпоинт эмбеддингов",
        "fl_embed_tip":    "Включает /v1/embeddings в API. Нужно только для создания векторов текста.",
        "kv_label":        "K/V Cache — можно выбрать разные типы для K и V (асимметрия рекомендуется для TurboQuant)",
        "kv_zero":         "0 = авто (рекомендуется). Полезно для NTK/YaRN.",
        "sys_prompt":      "Системный промпт",
        "sys_prompt_ph":   "Ты полезный помощник...",
        "extra_tip":       "Для будущих параметров: TurboQuant и т. д.",
        "extra_ph":        "--override-kv ...",
        "split_tip":       "Split Mode — распределение модели между GPU. 'layer' = по слоям (рекомендуется). 'row' = по строкам тензоров.",
        "tensor_tip":      "Tensor Split — доля видеопамяти каждой GPU (например: '1,1' = 50/50, '3,1' = 75/25). Пустое поле = авто.",
        "tensor_ph":       "1,1  (оставьте пустым для авто)",
        "disable_autofit": "Отключить auto-fit (-fit off)",
        "disable_autofit_tip": "Сервер самостоятельно пересчитывает распределение слоёв, чтобы оставить запас видеопамяти, и может проигнорировать Tensor Split. При включении этого параметра ngl/tensor-split строго соблюдаются — возможна нехватка видеопамяти (OOM).",
        "cpu_mode_tip":    "Режим: GPU (по умолчанию), только CPU+RAM, гибридный или Vulkan (AMD/Intel/NPU)",
        "cpu_tip2":        "Совет для CPU: для максимальной производительности установите Threads (раздел «Оборудование») равным числу физических ядер CPU.",
        "mmproj_tip":      "Файл mmproj для моделей с поддержкой зрения. Обнаруживается автоматически, если находится в папке модели.",
        "mmproj_none":     "mmproj не найден — у модели нет поддержки зрения или укажите файл вручную.",
        "mmproj_ph":       "Путь к файлу mmproj-*.gguf  (необязательно)",
        "mmproj_select":   "Выберите mmproj",
        "mmproj_disable":  "Отключить mmproj (экономит видеопамять, отключает зрение)",
        "mmproj_cpu":      "Загрузить mmproj в CPU/RAM вместо видеопамяти (--no-mmproj-offload)",

        # Botones modal / browser
        "load_btn":        "Загрузить  ↵",
        "cancel":          "Отмена",
        "delete_model":    "🗑️",
        "delete_model_confirm_title": "Удалить модель",
        "delete_model_confirm_msg":   "Удалить этот файл безвозвратно?\n\n{name}\n\n({size})",
        "delete_model_ok":  "Модель удалена.",
        "delete_model_err": "Ошибка удаления: {err}",
        "cleanup_backups_sec":   "🧹  Очистить старые резервные копии",
        "cleanup_backups_desc":  "Резервные копии предыдущих установок llama.cpp/TurboQuant, которые больше не нужны.",
        "cleanup_backups_scan":  "Найти резервные копии",
        "cleanup_backups_none":  "Резервные копии не найдены.",
        "cleanup_backups_found": "Найдено резервных копий: {n}  •  всего {size}",
        "cleanup_backups_del":   "Удалить все",
        "cleanup_backups_done":  "✓ Резервные копии удалены.",
        "change":          "Изменить",
        "manual_browse":   "📂  Выбрать файл вручную...",
        "folder_label":    "Папка:",
        "no_models_dir":   "В этой папке нет моделей.\n\nСкачайте модели на вкладке 🌐 Загрузка\nили измените папку выше.",
        "no_models_dir2":  "В этой папке нет моделей .gguf.\n\nСкачайте модель через 🌐 Загрузка или измените папку.",
        "my_models_title": "📂  Мои модели",
        "models_folder":   "Папка с моделями",
        "select_gguf":     "Выберите модель GGUF",

        # Mensajes de error / info
        "err_no_model":    "Сначала выберите модель.",
        "err_no_server":   "Откройте 'Сервер' и укажите путь к llama-server.exe",
        "err_not_found":   "Не найдено",
        "err_no_model_title":   "Модель не выбрана",
        "err_server_title":     "llama-server не найден",
        "err_title":            "Ошибка",
        "warn_stopped":         "Сервер остановлен",
        "warn_stopped_msg":     "Запустите сервер перед началом чата.",
        "warn_backend_title":   "Бэкенд не найден",
        "saved_title":          "Сохранено",
        "save_file_title":      "Сохранить созданный файл",
        "saved_msg":            "Файл сохранён в:\n",
        "err_save":             "Не удалось сохранить:\n",
        "err_read":             "Не удалось прочитать",
        "downloader_missing":   "llamastation_downloader.py не найден.\nПоместите файл в ту же папку.",
        "attach_image":         "Прикрепить изображение",
        "attach_files":         "Прикрепить файлы",
        "save_generated":       "Сохранить",
        "describe_image":       "Опиши это изображение.",
        "review_files":         "Изучи файлы.",

        # ── API Docs tab ──────────────────────────────────────────────
        "api_title":       "📡  Документация API",
        "api_subtitle":    "API llama.cpp, совместимый с OpenAI",
        "api_intro":       (
            "  Сервер llama.cpp предоставляет API, совместимый с OpenAI, по адресу:\n"
            "  http://127.0.0.1:{port}/v1\n\n"
            "  Приложения, использующие API OpenAI, могут подключаться по этому адресу."
        ),
        "api_sec_chat":    "💬  Ответы чата (Chat Completions)",
        "api_sec_stream":  "⚡  Потоковая передача (ответ в реальном времени)",
        "api_sec_models":  "📋  Модели",
        "api_sec_health":  "🩺  Состояние сервера",
        "api_sec_headless":"🖥️  Режим без интерфейса",
        "api_chat_desc":   "Создаёт ответ в чате",
        "api_stream_desc": "Поток SSE — токены в реальном времени",
        "api_models_desc": "Список загруженных моделей",
        "api_health_desc": "Проверяет готовность сервера",

        # ── Tema / idioma ─────────────────────────────────────────────
        "theme_to_light":  "☀  Светлая тема",
        "theme_to_dark":   "🌙  Тёмная тема",
        "lang_btn":        "🌐  English",

        # ── Actualización ─────────────────────────────────────────────
        "update_title":    "⬆  Обновить llama.cpp",
        "update_subtitle": "только официальный бэкенд · форки не затрагиваются",
        "update_close":    "Закрыть",
        "update_install":  "Установить обновление",
        "update_uptodate": "Установлена последняя версия",
        "update_installing":"Установка...",
        "update_done":     "✓ Установлено",
        "update_retry":    "Повторить",

        # Servidor — status
        "srv_running":     "Работает",
        "srv_running_ext": "Работает (внешний)",
        "srv_starting":    "Запуск...",
        "srv_port":        "порт: {port}",
        "srv_stopped_log": "Сервер остановлен. Видеопамять освобождена.",

        # ── Watchdog ─────────────────────────────────────────────────
        "watchdog_sec":           "🛡️  Мониторинг сервера",
        "watchdog_auto_relaunch": "Автоматически перезапускать сервер при сбое",
        "watchdog_crashed_title": "⚠️  Сбой сервера",
        "watchdog_crashed_msg":   "Сервер неожиданно остановился (код {rc}).\n\nПоследние строки журнала:\n\n{log}\n\nПодробнее см. вкладку «Журнал».",
        "watchdog_relaunch_log":  "🔄 Мониторинг: перезапуск сервера через {delay} с...",
        "watchdog_relaunching":   "Перезапуск...",

        # ── Sonido ───────────────────────────────────────────────────
        "sound_on":        "🔔  Звук ВКЛ",
        "sound_off":       "🔕  Звук ВЫКЛ",

        # ── Downloader ───────────────────────────────────────────────
        "dl_title":        "🌐  Скачать модели",
        "dl_save_in":      "Сохранить в:",
        "dl_search_ph":    "🔍  Поиск модели... (например: llama-3, qwen2.5, mistral)",
        "dl_search_btn":   "Найти",
        "dl_arch":         "Архитектура:",
        "dl_author":       "Автор:",
        "dl_searching":    "Поиск...",
        "dl_no_results":   "Модели не найдены.\nПопробуйте другой запрос.",
        "dl_start_hint":   "Найдите модель, чтобы начать,\nили просмотрите популярные →",
        "dl_select_hint":  "← Выберите модель из списка",
        "dl_files_title":  "ДОСТУПНЫЕ ФАЙЛЫ",
        "dl_quant_filter": "Фильтр квантования:",
        "dl_files_ph":     "Выберите модель для просмотра файлов",
        "dl_loading":      "Загрузка файлов...",
        "dl_no_files":     "Нет файлов, соответствующих фильтру",
        "dl_downloads":    "АКТИВНЫЕ ЗАГРУЗКИ",
        "dl_no_downloads": "Нет активных загрузок",
        "dl_view_hf":      "Открыть на HF ↗",
        "dl_downloads_lbl":"загрузок",
        "dl_updated":      "Обновлено:",
        "dl_caps_warning": "⚠ Указанные возможности могут быть неполными",

        # ── MTP ──────────────────────────────────────────────────────
        "mtp_activate":    "Включить MTP",
        "mtp_manual_hint": (
            "💡 Укажите точные параметры, необходимые вашему бэкенду (по умолчанию "
            "они не заданы). Они применяются только при включённом MTP, который "
            "также автоматически задаёт -np 1.\n"
            "Например, AtomicChat: --spec-type nextn --draft-max 4"
        ),
    },

    "en": {
        # ── Navigation ────────────────────────────────────────────────
        "nav_chat":        "Chat",
        "nav_agent":       "Agent",
        "nav_server":      "Server",
        "nav_logs":        "Logs",
        "nav_info":        "Model Info",
        "nav_download":    "Download",
        "nav_api":         "API Docs",
        "nav_voice":       "Voice",
        "nav_about":       "About",

        # ── Right sidebar ─────────────────────────────────────────────
        "loaded_model":    "LOADED MODEL",
        "no_model":        "None",
        "my_models":       "📂  My models",
        "download_models": "🌐  Download models",
        "server_label":    "SERVER",
        "server_b_label":  "SERVER B",
        "server_stopped":  "Stopped",
        "server_port":     "port: —",
        "start_server":    "▶  Start server",
        "stop_server":     "■  Stop",
        "backend_label":   "BACKEND",
        "backend_b_label": "BACKEND B",
        "update_llama":    "⬆  Update / Download backend",
        "controls":        "Controls",

        # ── Left sidebar ──────────────────────────────────────────────
        "app_title":       "⚡ LlamaStation",
        "conversations":   "CONVERSATIONS",
        "new_chat":        "＋  New chat",
        "no_history":      "No conversations yet\nStart chatting →",

        # ── Chat ──────────────────────────────────────────────────────
        "chat_title":      "Chat",
        "clear_chat":      "Clear",
        "thinking_on":         "👁 Razonamiento",
        "thinking_off":        "👁 Razonamiento OFF",
        "enable_thinking_on":  "🧠 Think: ON",
        "enable_thinking_off": "⚡ Think: OFF",
        "web_on":          "🌐 Web ON",
        "web_off":         "🌐 Web OFF",
        "system_label":    "System:",
        "system_ph":       "System prompt...",
        "send":            "Send",
        "sending":         "...",
        "you":             "You",
        "model":           "Model",

        # ── Server tab ────────────────────────────────────────────────
        "server_tab_title":   "Server Configuration",
        "server_exe":         "🔧 llama-server executable",
        "server_network":     "🌐 Network",
        "server_port_lbl":    "Port",
        "server_host_lbl":    "Host",
        "server_port_b_lbl":  "Port B (9090)",
        "server_save":        "💾 Save",
        "server_cmd":         "📋 Generated command",
        "server_cmd_preview": "Update preview",
        "server_saved_ok":    "Configuration saved.",
        "browse":             "Browse...",
        "server_headless_sec": "🖥️  Headless Mode",
        "server_headless_desc": (
            "Start LlamaStation without the graphical interface.\n"
            "Uses the saved profile of the selected model."
        ),
        "server_headless_cmd_label": "Command (copy and paste into your terminal):",
        "server_headless_no_model":  "← Load a model first",
        "copy":               "📋 Copy",

        # ── Logs tab ──────────────────────────────────────────────────
        "logs_title":      "Server logs",
        "logs_clear":      "Clear",

        # ── Info tab ──────────────────────────────────────────────────
        "info_title":      "Model info",
        "info_no_server":  "⚠ Server is not running.",

        # ── Load model modal ──────────────────────────────────────────
        "modal_vram":      "Estimated VRAM usage",
        "modal_vram_hint": "depends on model and ctx",
        "modal_cancel":    "Cancel",
        "modal_load":      "Load Model  ↵",
        "sec_hardware":    "Hardware",
        "sec_multigpu":    "Multi-GPU  (2x RTX 3060)",
        "sec_cpu":         "CPU / NPU — No dedicated GPU",
        "sec_context":     "Context & Batch",
        "sec_sampling":    "Sampling",
        "sec_flags":       "Advanced Options",
        "sec_rope":        "RoPE Scaling  (context extension)",
        "sec_vision":      "Vision (mmproj)",
        "sec_extra":       "Extra Arguments (free text)",

        # Sliders
        "sl_gpu_layers":   "GPU Layers  — model layers on GPU (-1 = all, reduce if running out of VRAM)",
        "sl_threads":      "Threads  — CPU threads for the non-GPU part",
        "sl_threads_b":    "Threads Batch  — CPU threads for initial prompt processing",
        "sl_max_conc":     "Max Concurrent  — simultaneous requests, experimental, keep at 1",
        "sl_ctx":          "Context Size  — tokens the model remembers, more = more VRAM",
        "sl_batch":        "Batch Size  — tokens processed at once when loading prompt, more = faster",
        "sl_ubatch":       "UBatch Size  — batch subdivision, must be equal or less than Batch Size",
        "sl_max_tok":      "Max Tokens  — generation token limit per response (-1 = unlimited)",
        "sl_ctx_checkpoints": "Ctx Checkpoints  — context-state checkpoints kept (def. 32)",
        "sl_temp":         "Temperature  — creativity: 0=deterministic, 0.7=balanced, >1=more random",
        "sl_topk":         "Top K  — limits sampling to the K most probable tokens, 0=disabled",
        "sl_topp":         "Top P  — nucleus sampling: uses tokens until this cumulative probability",
        "sl_minp":         "Min P  — discards tokens with probability below this % of top token",
        "sl_rep_pen":      "Repeat Penalty  — penalizes word repetition, 1.0=no penalty",
        "sl_rep_n":        "Repeat Last N  — how many tokens back to look for Repeat Penalty",
        "sl_seed":         "Seed  — random seed, -1=different each time, set to reproduce results",
        "sl_rope_base":    "RoPE Freq Base  — base frequency for context scaling, 0=Auto",
        "sl_rope_scale":   "RoPE Freq Scale  — frequency scaling factor, 0=Auto",

        # Flags
        "fl_flash":        "Flash Attention",
        "fl_flash_tip":    "Speeds up attention computation on GPU. Always recommended.",
        "fl_mmap":         "mmap()",
        "fl_mmap_tip":     "Memory-map: loads the model by mapping the file into memory. More efficient.",
        "fl_mlock":        "mlock",
        "fl_mlock_tip":    "Locks the model in RAM so the OS won't swap it out.",
        "fl_contbatch":    "Continuous Batching",
        "fl_contbatch_tip":"Handles multiple requests simultaneously without waiting. Recommended.",
        "fl_kvoff":        "KV Cache Offload",
        "fl_kvoff_tip":    "Moves the KV cache to VRAM. Allows longer contexts.",
        "fl_keepmem":      "Keep In Memory",
        "fl_keepmem_tip":  "Doesn't unload the model when stopping the server. Faster next start.",
        "fl_embed":        "Embeddings endpoint",
        "fl_embed_tip":    "Enables /v1/embeddings in the API. Only needed for text vector generation.",
        "kv_label":        "K/V Cache — you can choose different types for K and V (asymmetry recommended with TurboQuant)",
        "kv_zero":         "0 = Auto (recommended). Useful for NTK/YaRN.",
        "sys_prompt":      "System Prompt",
        "sys_prompt_ph":   "You are a helpful assistant...",
        "extra_tip":       "For future flags: TurboQuant, etc.",
        "extra_ph":        "--override-kv ...",
        "split_tip":       "Split Mode — how the model is split across GPUs. 'layer' = layer split (recommended). 'row' = tensor row split.",
        "tensor_tip":      "Tensor Split — VRAM ratio per GPU (e.g. '1,1' = 50/50, '3,1' = 75/25). Empty = auto.",
        "tensor_ph":       "1,1  (leave empty for auto)",
        "disable_autofit": "Disable auto-fit (-fit off)",
        "disable_autofit_tip": "The server recalculates the layer split on its own to leave a VRAM safety margin, and may ignore your Tensor Split. With this forced, it honors ngl/tensor-split exactly — risk of OOM if you overshoot VRAM.",
        "cpu_mode_tip":    "Mode: GPU (default), CPU+RAM only, Hybrid, or Vulkan (AMD/Intel/NPU)",
        "cpu_tip2":        "CPU tip: raise Threads (Hardware section) to your CPU's physical core count for best performance.",
        "mmproj_tip":      "mmproj file for vision models. Auto-detected if it's in the same folder as the model.",
        "mmproj_none":     "No mmproj detected — this model has no vision, or set it manually.",
        "mmproj_ph":       "Path to mmproj-*.gguf file  (optional)",
        "mmproj_select":   "Select mmproj",
        "mmproj_disable":  "Disable mmproj (saves VRAM, disables vision)",
        "mmproj_cpu":      "Load mmproj on CPU/RAM instead of VRAM (--no-mmproj-offload)",

        # Buttons modal / browser
        "load_btn":        "Load  ↵",
        "cancel":          "Cancel",
        "delete_model":    "🗑️",
        "delete_model_confirm_title": "Delete model",
        "delete_model_confirm_msg":   "Permanently delete this file?\n\n{name}\n\n({size})",
        "delete_model_ok":  "Model deleted.",
        "delete_model_err": "Error deleting: {err}",
        "cleanup_backups_sec":   "🧹  Clean up old backups",
        "cleanup_backups_desc":  "Backups from previous llama.cpp/TurboQuant installations that are no longer needed.",
        "cleanup_backups_scan":  "Scan for backups",
        "cleanup_backups_none":  "No backups found.",
        "cleanup_backups_found": "{n} backup(s) found  •  {size} total",
        "cleanup_backups_del":   "Delete all",
        "cleanup_backups_done":  "✓ Backups deleted.",
        "change":          "Change",
        "manual_browse":   "📂  Browse file manually...",
        "folder_label":    "Folder:",
        "no_models_dir":   "No models in this folder.\n\nDownload models from the 🌐 Download tab\nor change the folder above.",
        "no_models_dir2":  "No .gguf models in this folder.\n\nDownload from 🌐 Download or change the folder.",
        "my_models_title": "📂  My models",
        "models_folder":   "Models folder",
        "select_gguf":     "Select GGUF model",

        # Error / info messages
        "err_no_model":    "Select a model first.",
        "err_no_server":   "Go to 'Server' and specify the path to llama-server.exe",
        "err_not_found":   "Not found",
        "err_no_model_title":   "No model",
        "err_server_title":     "llama-server not found",
        "err_title":            "Error",
        "warn_stopped":         "Server stopped",
        "warn_stopped_msg":     "Start the server before chatting.",
        "warn_backend_title":   "Backend not found",
        "saved_title":          "Saved",
        "save_file_title":      "Save generated file",
        "saved_msg":            "File saved at:\n",
        "err_save":             "Could not save:\n",
        "err_read":             "Could not read",
        "downloader_missing":   "llamastation_downloader.py not found.\nPlace the file in the same folder.",
        "attach_image":         "Attach image",
        "attach_files":         "Attach file(s)",
        "save_generated":       "Save",
        "describe_image":       "Describe this image.",
        "review_files":         "Review the files.",

        # ── API Docs tab ──────────────────────────────────────────────
        "api_title":       "📡  API Docs",
        "api_subtitle":    "llama.cpp OpenAI-compatible API",
        "api_intro":       (
            "  Your llama.cpp server exposes an OpenAI-compatible API at:\n"
            "  http://127.0.0.1:{port}/v1\n\n"
            "  Any app that uses the OpenAI API works by pointing it to this URL."
        ),
        "api_sec_chat":    "💬  Chat Completions",
        "api_sec_stream":  "⚡  Streaming (real-time response)",
        "api_sec_models":  "📋  Models",
        "api_sec_health":  "🩺  Server status",
        "api_sec_headless":"🖥️  Headless Mode (no window)",
        "api_chat_desc":   "Generate a chat response",
        "api_stream_desc": "SSE Streaming — real-time tokens",
        "api_models_desc": "List loaded models",
        "api_health_desc": "Check if the server is ready",

        # ── Theme / language ──────────────────────────────────────────
        "theme_to_light":  "☀  Light mode",
        "theme_to_dark":   "🌙  Dark mode",
        "lang_btn":        "🌐  Русский",

        # ── Update ────────────────────────────────────────────────────
        "update_title":    "⬆  Update llama.cpp",
        "update_subtitle": "official backend only · forks not affected",
        "update_close":    "Close",
        "update_install":  "Install update",
        "update_uptodate": "Already up to date",
        "update_installing":"Installing...",
        "update_done":     "✓ Installed",
        "update_retry":    "Retry",

        # Server — status
        "srv_running":     "Running",
        "srv_running_ext": "Running (external)",
        "srv_starting":    "Starting...",
        "srv_port":        "port: {port}",
        "srv_stopped_log": "Server stopped. VRAM freed.",

        # ── Watchdog ─────────────────────────────────────────────────
        "watchdog_sec":           "🛡️  Server Watchdog",
        "watchdog_auto_relaunch": "Automatically restart server if it crashes",
        "watchdog_crashed_title": "⚠️  Server crashed",
        "watchdog_crashed_msg":   "The server stopped unexpectedly (code {rc}).\n\nLast log lines:\n\n{log}\n\nCheck the Logs tab for details.",
        "watchdog_relaunch_log":  "🔄 Watchdog: restarting server in {delay}s...",
        "watchdog_relaunching":   "Restarting...",

        # ── Sound ────────────────────────────────────────────────────
        "sound_on":        "🔔  Sound ON",
        "sound_off":       "🔕  Sound OFF",

        # ── Downloader ───────────────────────────────────────────────
        "dl_title":        "🌐  Download Models",
        "dl_save_in":      "Save to:",
        "dl_search_ph":    "🔍  Search model... (e.g. llama-3, qwen2.5, mistral)",
        "dl_search_btn":   "Search",
        "dl_arch":         "Architecture:",
        "dl_author":       "Author:",
        "dl_searching":    "Searching...",
        "dl_no_results":   "No models found.\nTry a different search.",
        "dl_start_hint":   "Search for a model to get started\nor browse the most popular →",
        "dl_select_hint":  "← Select a model from the list",
        "dl_files_title":  "AVAILABLE FILES",
        "dl_quant_filter": "Filter quant:",
        "dl_files_ph":     "Select a model to see its files",
        "dl_loading":      "Loading files...",
        "dl_no_files":     "No files match this filter",
        "dl_downloads":    "ACTIVE DOWNLOADS",
        "dl_no_downloads": "No active downloads",
        "dl_view_hf":      "View on HF ↗",
        "dl_downloads_lbl":"downloads",
        "dl_updated":      "Updated:",
        "dl_caps_warning": "⚠ Shown capabilities may be incomplete",

        # ── MTP ──────────────────────────────────────────────────────
        "mtp_activate":    "Enable MTP",
        "mtp_manual_hint": (
            "💡 Type the exact flags your backend needs here (nothing is preloaded). "
            "They're only applied while the MTP toggle is on, which also forces -np 1 "
            "automatically.\n"
            "E.g. AtomicChat: --spec-type nextn --draft-max 4"
        ),
    },
}

# Idioma activo — se actualiza desde LlamaStation al cambiar
_LANG = "es"

def set_lang(lang: str):
    global _LANG
    if lang in STRINGS:
        _LANG = lang

def get_lang() -> str:
    return _LANG

def T(key: str, **kwargs) -> str:
    """Devuelve el string traducido. kwargs para interpolación: T('srv_port', port=8080)"""
    s = STRINGS.get(_LANG, STRINGS["es"]).get(key)
    if s is None:
        # Fallback al español
        s = STRINGS["es"].get(key, key)
    if kwargs:
        try:
            return s.format(**kwargs)
        except Exception:
            return s
    return s

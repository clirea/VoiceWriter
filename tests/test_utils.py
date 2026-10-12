import json
import os
from unittest.mock import patch

import pytest

from utils import load_config, mask_api_key, save_config


@pytest.fixture
def config_path(tmp_path):
    return str(tmp_path / "voicewriter_config.json")


class TestLoadConfig:
    def test正常系_load_configメソッドが保存済みの設定を返すこと(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"shortcutKey": "ctrl+space"}, f)

        assert load_config(config_path) == {"shortcutKey": "ctrl+space"}

    def test正常系_load_configメソッドがファイルがない場合に空の設定を返すこと(self, config_path):
        assert load_config(config_path) == {}

    def test正常系_load_configメソッドが旧版のASCIIエスケープされた設定を読めること(self, config_path):
        with open(config_path, "w", encoding="ascii") as f:
            json.dump({"selectedMic": "マイク"}, f, ensure_ascii=True)

        assert load_config(config_path) == {"selectedMic": "マイク"}

    def test異常系_load_configメソッドが壊れたJSONで例外を出さず空の設定を返すこと(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{broken")

        assert load_config(config_path) == {}

    def test異常系_load_configメソッドが壊れたJSONを退避すること(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{broken")

        load_config(config_path)

        with open(config_path + ".broken", encoding="utf-8") as f:
            assert f.read() == "{broken"

    def test異常系_load_configメソッドが再び壊れたときに過去の退避を上書きしないこと(self, config_path):
        with open(config_path + ".broken", "w", encoding="utf-8") as f:
            f.write("{first")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{second")

        load_config(config_path)

        with open(config_path + ".broken", encoding="utf-8") as f:
            assert f.read() == "{first"

    def test異常系_load_configメソッドが再び壊れた設定を連番の退避先へ残すこと(self, config_path):
        with open(config_path + ".broken", "w", encoding="utf-8") as f:
            f.write("{first")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{second")

        load_config(config_path)

        with open(config_path + ".broken.1", encoding="utf-8") as f:
            assert f.read() == "{second"

    def test異常系_load_configメソッドが三度目に壊れた設定を次の連番へ残すこと(self, config_path):
        for suffix, content in ((".broken", "{first"), (".broken.1", "{second"), ("", "{third")):
            with open(config_path + suffix, "w", encoding="utf-8") as f:
                f.write(content)

        load_config(config_path)

        with open(config_path + ".broken.2", encoding="utf-8") as f:
            assert f.read() == "{third"

    def test異常系_load_configメソッドが退避できない場合も空の設定を返すこと(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{broken")

        with patch("utils.os.replace", side_effect=OSError("locked")):
            assert load_config(config_path) == {}

    def test異常系_load_configメソッドが退避できない場合に空の退避ファイルを残さないこと(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("{broken")

        with patch("utils.os.replace", side_effect=OSError("locked")):
            load_config(config_path)

        assert not os.path.exists(config_path + ".broken")

    def test異常系_load_configメソッドがオブジェクト以外のJSONで空の設定を返すこと(self, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(["not", "a", "dict"], f)

        assert load_config(config_path) == {}

    def test異常系_load_configメソッドがUTF8でないファイルで空の設定を返すこと(self, config_path):
        with open(config_path, "wb") as f:
            f.write('{"selectedMic": "マイク"}'.encode("cp932"))

        assert load_config(config_path) == {}


class TestSaveConfig:
    def test正常系_save_configメソッドが日本語を含む設定を読み戻せる形で保存すること(self, config_path):
        save_config({"selectedMic": "マイク (USB)"}, config_path)

        assert load_config(config_path) == {"selectedMic": "マイク (USB)"}

    def test正常系_save_configメソッドが既存の設定を置き換えること(self, config_path):
        save_config({"shortcutKey": "ctrl+space"}, config_path)

        save_config({"shortcutKey": "f9"}, config_path)

        assert load_config(config_path) == {"shortcutKey": "f9"}

    def test正常系_save_configメソッドが作業用ファイルを残さないこと(self, config_path):
        save_config({"shortcutKey": "f9"}, config_path)

        assert os.listdir(os.path.dirname(config_path)) == ["voicewriter_config.json"]

    def test異常系_save_configメソッドが書き込めない値でTypeErrorを送出すること(self, config_path):
        with pytest.raises(TypeError):
            save_config({"value": object()}, config_path)

    def test異常系_save_configメソッドが失敗時に既存の設定を保持すること(self, config_path):
        save_config({"shortcutKey": "ctrl+space"}, config_path)

        with pytest.raises(TypeError):
            save_config({"value": object()}, config_path)

        assert load_config(config_path) == {"shortcutKey": "ctrl+space"}

    def test異常系_save_configメソッドが失敗時に作業用ファイルを残さないこと(self, config_path):
        with pytest.raises(TypeError):
            save_config({"value": object()}, config_path)

        assert os.listdir(os.path.dirname(config_path)) == []


class TestMaskApiKey:
    def test正常系_mask_api_keyメソッドが先頭と末尾の4文字だけを残すこと(self):
        assert mask_api_key("sk-abcdefghijklmnop") == "sk-a***********mnop"

    def test正常系_mask_api_keyメソッドが8文字以下のキーをすべて伏せること(self):
        assert mask_api_key("sk-12345") == "********"

    def test正常系_mask_api_keyメソッドが空文字で空文字を返すこと(self):
        assert mask_api_key("") == ""

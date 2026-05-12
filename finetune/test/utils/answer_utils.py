"""Answer extraction and cleaning utilities"""
import re
from typing import Optional


class UnitTextManager:
    """管理单位文本，用于清理答案中的单位"""
    def __init__(self):
        self.unit_texts = [
            "east", "degree", "mph", "kmph", "ft", "m sqaure", "m east", "sq m", "deg", "mile",
            "ratio", "profit of rs", "rd", "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt",
            "gain", "ab", "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .",
            "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess", "st", "inches",
            "noon", "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th", "π r 2",
            "sq", "mark", "l", "toy", "coin", "sq . m", "gallon", "° f", "profit", "minw", "yr",
            "women", "feet", "am", "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee",
            "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day", "hour", "minute",
            "min", "second", "man", "woman", "sec", "cube", "mt", "sq inch", "mp", "∏ cm ³",
            "hectare", "more", "sec", "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month",
            "km", "m", "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year", "increase",
            "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .", "metre", "meter", "inch",
        ]
        self.unit_texts.extend([t + "s" for t in self.unit_texts])

    def clean_units(self, string: str) -> str:
        """清理字符串中的单位"""
        for unit_text in self.unit_texts:
            string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        return string
    

class StringCleaner:
    """字符串清理器"""
    def __init__(self, unit_manager: UnitTextManager):
        self.unit_manager = unit_manager

    def _fix_fracs(self, string: str) -> str:
        """修复分数表达式"""
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            for substr in substrs[1:]:
                new_str += "\\frac"
                if len(substr) > 0 and substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) >= 2:
                        a, b = substr[0], substr[1]
                        if b != "{":
                            new_str += f"{{{a}}}{{{b}}}{substr[2:]}" if len(substr) > 2 else f"{{{a}}}{{{b}}}"
                        else:
                            new_str += f"{{{a}}}{b}{substr[2:]}" if len(substr) > 2 else f"{{{a}}}{b}"
                    else:
                        return string
        return new_str

    def _fix_a_slash_b(self, string: str) -> str:
        """修复 a/b 格式的分数"""
        if len(string.split("/")) != 2:
            return string
        a, b = string.split("/")
        try:
            a = int(a) if "sqrt" not in a else a
            b = int(b) if "sqrt" not in b else b
            assert string == f"{a}/{b}"
            return f"\\frac{{{a}}}{{{b}}}"
        except:
            return string

    def _fix_sqrt(self, string: str) -> str:
        """修复平方根表达式"""
        return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)

    def strip_string(self, string: str, skip_unit: bool = False) -> str:
        """清理字符串"""
        string = str(string).strip()
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = string.replace("$", "")
        string = string.replace("\\text", "")
        string = string.replace("x\\in", "")
        string = string.replace("\\mathrm", "")

        string = self._fix_fracs(string)
        string = self._fix_a_slash_b(string)
        string = self._fix_sqrt(string)
        string = string.replace(" ", "")

        if not skip_unit:
            string = self.unit_manager.clean_units(string)

        string = re.sub(r"(\\text\{)(.*?)(\})", "\\2", string)
        string = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", string)
        string = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", string)
        string = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", string)

        # Convert \frac{X}{Y} to (X)/(Y) and \sqrt{X} to sqrt(X) before generic brace removal
        string = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", string)
        string = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", string)

        string = re.sub(r"(\{)(.*?)(\})", "\\2", string)
        string = string.strip()

        if string.startswith("\\") and len(string) > 1:
            string = string[1:]

        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        string = string.replace("%", "")
        string = string.replace(" .", ".")

        if len(string) > 0 and string[0] == ".":
            string = "0" + string

        if len(string) > 0 and string[-1] == ".":
            string = string[:-1]

        if string.startswith("{") and string.endswith("}"):
            string = string[1:-1]

        string = string.replace(",", "")
        string = string.replace("\\,", "")

        return string


class AnswerExtractor:
    """答案提取器"""
    def __init__(self, string_cleaner: StringCleaner):
        self.string_cleaner = string_cleaner

    def extract_answer(self, pred_str: str, use_last_number: bool = True) -> str:
        """从预测字符串中提取最终答案"""
        if not pred_str:
            return ""

        pred_str = str(pred_str).replace("\u043a\u0438", "")

        # 检测 boxed 格式
        if "boxed" in pred_str.lower():
            return self._extract_boxed_answer(pred_str)

        # 检测 "the answer is" 模式
        if "the answer is" in pred_str.lower():
            pred = pred_str.lower().split("the answer is")[-1].strip()
            return self.string_cleaner.strip_string(pred)

        # 优先提取数学表达式
        if "\\frac" in pred_str or "\\sqrt" in pred_str:
            if "$" in pred_str:
                parts = pred_str.split("$")
                for part in reversed(parts):
                    cleaned = self.string_cleaner.strip_string(part)
                    if cleaned and ("frac" in cleaned or "sqrt" in cleaned):
                        return cleaned

        # 提取最后一个数字
        if use_last_number:
            return self._get_last_number_answer(pred_str)

        return "" 

    def _extract_boxed_answer(self, pred_str: str) -> str:
        """提取 boxed 格式的答案"""
        ans = pred_str.lower().split("boxed")[-1]
        if ans.startswith("{"):
            return self._extract_bracketed_answer(ans)
        else:
            ans = ans.split("$")[0].strip()
            return self.string_cleaner.strip_string(ans)

    def _extract_bracketed_answer(self, ans: str) -> str:
        """提取括号中的答案"""
        stack = 1
        result = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                result += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                result += c
            else:
                result += c
        return self.string_cleaner.strip_string(result)

    def _get_last_number_answer(self, pred_str: str) -> str:
        """提取最后一个数字"""
        pattern = r"-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if pred:
            return self.string_cleaner.strip_string(pred[-1])
        return ""

"""
时间工具类

支持年月日的加减操作，能够处理多种时间格式：
- 完整日期: 2002-09-15
- 年月: 2002-09
- 仅年份: 2002
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from enum import Enum


class TimeGranularity(str, Enum):
    """时间粒度"""
    YEAR = "year"       # 仅年份
    MONTH = "month"     # 年月
    DAY = "day"         # 年月日


@dataclass
class TemporalDate:
    """
    时间日期类，支持不同粒度的时间表示和加减操作
    
    Examples:
        >>> t = TemporalDate.parse("2002-09")
        >>> t.add_years(-1)
        TemporalDate(year=2001, month=9, day=None)
        >>> str(t.add_years(-1))
        '2001-09'
    """
    year: int
    month: Optional[int] = None
    day: Optional[int] = None
    
    @property
    def granularity(self) -> TimeGranularity:
        """获取时间粒度"""
        if self.day is not None:
            return TimeGranularity.DAY
        elif self.month is not None:
            return TimeGranularity.MONTH
        else:
            return TimeGranularity.YEAR
    
    @classmethod
    def parse(cls, date_str: str) -> "TemporalDate":
        """
        解析时间字符串
        
        支持格式:
        - YYYY: 2002
        - YYYY-MM: 2002-09
        - YYYY-MM-DD: 2002-09-15
        - YYYY/MM/DD: 2002/09/15
        - YYYYMMDD: 20020915
        
        Args:
            date_str: 时间字符串
            
        Returns:
            TemporalDate 对象
            
        Raises:
            ValueError: 无法解析的时间格式
        """
        date_str = date_str.strip()
        
        # 尝试匹配 YYYY-MM-DD 或 YYYY/MM/DD
        match = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', date_str)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            cls._validate_date(year, month, day)
            return cls(year=year, month=month, day=day)
        
        # 尝试匹配 YYYY-MM 或 YYYY/MM
        match = re.match(r'^(\d{4})[-/](\d{1,2})$', date_str)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            cls._validate_date(year, month)
            return cls(year=year, month=month)
        
        # 尝试匹配 YYYYMMDD
        match = re.match(r'^(\d{4})(\d{2})(\d{2})$', date_str)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            cls._validate_date(year, month, day)
            return cls(year=year, month=month, day=day)
        
        # 尝试匹配 YYYYMM
        match = re.match(r'^(\d{4})(\d{2})$', date_str)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            cls._validate_date(year, month)
            return cls(year=year, month=month)
        
        # 尝试匹配 YYYY
        match = re.match(r'^(\d{4})$', date_str)
        if match:
            year = int(match.group(1))
            return cls(year=year)
        
        raise ValueError(f"无法解析时间格式: {date_str}")
    
    @staticmethod
    def _validate_date(year: int, month: Optional[int] = None, day: Optional[int] = None) -> None:
        """验证日期有效性"""
        if year < 1 or year > 9999:
            raise ValueError(f"年份超出范围: {year}")
        
        if month is not None:
            if month < 1 or month > 12:
                raise ValueError(f"月份无效: {month}")
            
            if day is not None:
                max_day = TemporalDate._days_in_month(year, month)
                if day < 1 or day > max_day:
                    raise ValueError(f"日期无效: {year}-{month:02d}-{day:02d}")
    
    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """判断是否为闰年"""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        """获取某月的天数"""
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if month == 2 and TemporalDate._is_leap_year(year):
            return 29
        return days[month - 1]
    
    def add_years(self, years: int) -> "TemporalDate":
        """
        加减年份
        
        Args:
            years: 年份增量（正数为加，负数为减）
            
        Returns:
            新的 TemporalDate 对象
            
        Examples:
            >>> TemporalDate.parse("2002-09").add_years(-1)
            TemporalDate(year=2001, month=9, day=None)
        """
        new_year = self.year + years
        new_day = self.day
        
        # 处理闰年 2月29日 的情况
        if self.month == 2 and self.day == 29:
            if not self._is_leap_year(new_year):
                new_day = 28
        
        return TemporalDate(year=new_year, month=self.month, day=new_day)
    
    def add_months(self, months: int) -> "TemporalDate":
        """
        加减月份
        
        Args:
            months: 月份增量（正数为加，负数为减）
            
        Returns:
            新的 TemporalDate 对象
            
        Examples:
            >>> TemporalDate.parse("2002-09-15").add_months(-2)
            TemporalDate(year=2002, month=7, day=15)
            >>> TemporalDate.parse("2002-01").add_months(-2)
            TemporalDate(year=2001, month=11, day=None)
        """
        if self.month is None:
            raise ValueError("无法对仅包含年份的日期进行月份加减")
        
        # 计算总月数
        total_months = self.year * 12 + (self.month - 1) + months
        new_year = total_months // 12
        new_month = (total_months % 12) + 1
        
        # 处理日期溢出（如 1月31日 加1个月 -> 2月28日）
        new_day = self.day
        if self.day is not None:
            max_day = self._days_in_month(new_year, new_month)
            if self.day > max_day:
                new_day = max_day
        
        return TemporalDate(year=new_year, month=new_month, day=new_day)
    
    def add_days(self, days: int) -> "TemporalDate":
        """
        加减天数
        
        Args:
            days: 天数增量（正数为加，负数为减）
            
        Returns:
            新的 TemporalDate 对象
            
        Examples:
            >>> TemporalDate.parse("2002-09-15").add_days(20)
            TemporalDate(year=2002, month=10, day=5)
        """
        if self.day is None:
            raise ValueError("无法对不包含日期的时间进行天数加减")
        
        # 转换为天数进行计算
        total_days = self._to_days() + days
        return self._from_days(total_days)
    
    def _to_days(self) -> int:
        """将日期转换为从公元1年1月1日起的天数"""
        year = self.year
        month = self.month or 1
        day = self.day or 1
        
        # 简化计算：使用格里高利历
        # 计算年份贡献的天数
        y = year - 1
        days = y * 365 + y // 4 - y // 100 + y // 400
        
        # 计算月份贡献的天数
        days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        days += days_before_month[month - 1]
        
        # 闰年且月份大于2月
        if month > 2 and self._is_leap_year(year):
            days += 1
        
        # 加上日
        days += day
        
        return days
    
    @classmethod
    def _from_days(cls, total_days: int) -> "TemporalDate":
        """从天数转换为日期"""
        # 估算年份
        year = total_days // 366
        while True:
            days_in_year = 366 if cls._is_leap_year(year) else 365
            year_start = cls(year=year, month=1, day=1)._to_days()
            if year_start > total_days:
                year -= 1
            elif year_start + days_in_year <= total_days:
                year += 1
            else:
                break
        
        # 计算月和日
        remaining = total_days - cls(year=year, month=1, day=1)._to_days() + 1
        
        for month in range(1, 13):
            days_in_month = cls._days_in_month(year, month)
            if remaining <= days_in_month:
                return cls(year=year, month=month, day=remaining)
            remaining -= days_in_month
        
        # 不应该到达这里
        raise ValueError(f"日期计算错误: total_days={total_days}")
    
    def subtract(self, other: "TemporalDate") -> Tuple[int, int, int]:
        """
        计算两个日期之间的差值
        
        Returns:
            (年差, 月差, 日差) 的元组
            
        Note:
            只有相同粒度的日期才能精确计算差值
        """
        years = self.year - other.year
        months = 0
        days = 0
        
        if self.month is not None and other.month is not None:
            months = self.month - other.month
            
        if self.day is not None and other.day is not None:
            days = self.day - other.day
        
        return (years, months, days)
    
    def __str__(self) -> str:
        """转换为字符串"""
        if self.day is not None:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        elif self.month is not None:
            return f"{self.year:04d}-{self.month:02d}"
        else:
            return f"{self.year:04d}"
    
    def __repr__(self) -> str:
        return f"TemporalDate(year={self.year}, month={self.month}, day={self.day})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporalDate):
            return NotImplemented
        return self.year == other.year and self.month == other.month and self.day == other.day
    
    def __lt__(self, other: "TemporalDate") -> bool:
        """小于比较"""
        if self.year != other.year:
            return self.year < other.year
        
        # 处理 None 的情况
        self_month = self.month or 0
        other_month = other.month or 0
        if self_month != other_month:
            return self_month < other_month
        
        self_day = self.day or 0
        other_day = other.day or 0
        return self_day < other_day
    
    def __le__(self, other: "TemporalDate") -> bool:
        return self == other or self < other
    
    def __gt__(self, other: "TemporalDate") -> bool:
        return not self <= other
    
    def __ge__(self, other: "TemporalDate") -> bool:
        return not self < other
    
    def __hash__(self) -> int:
        return hash((self.year, self.month, self.day))
    
    def to_tuple(self) -> Tuple[int, Optional[int], Optional[int]]:
        """转换为元组 (year, month, day)"""
        return (self.year, self.month, self.day)
    
    def format(self, fmt: str = "%Y-%m-%d") -> str:
        """
        格式化输出
        
        支持的格式符:
        - %Y: 四位年份
        - %y: 两位年份
        - %m: 两位月份
        - %d: 两位日期
        
        Args:
            fmt: 格式字符串
            
        Returns:
            格式化后的字符串
        """
        result = fmt
        result = result.replace("%Y", f"{self.year:04d}")
        result = result.replace("%y", f"{self.year % 100:02d}")
        if self.month is not None:
            result = result.replace("%m", f"{self.month:02d}")
        if self.day is not None:
            result = result.replace("%d", f"{self.day:02d}")
        return result


def add_years(date_str: str, years: int) -> str:
    """
    对日期字符串加减年份
    
    Args:
        date_str: 日期字符串
        years: 年份增量
        
    Returns:
        计算后的日期字符串
        
    Examples:
        >>> add_years("2002-09", -1)
        '2001-09'
    """
    return str(TemporalDate.parse(date_str).add_years(years))


def add_months(date_str: str, months: int) -> str:
    """
    对日期字符串加减月份
    
    Args:
        date_str: 日期字符串
        months: 月份增量
        
    Returns:
        计算后的日期字符串
        
    Examples:
        >>> add_months("2002-09-15", -2)
        '2002-07-15'
    """
    return str(TemporalDate.parse(date_str).add_months(months))


def add_days(date_str: str, days: int) -> str:
    """
    对日期字符串加减天数
    
    Args:
        date_str: 日期字符串
        days: 天数增量
        
    Returns:
        计算后的日期字符串
        
    Examples:
        >>> add_days("2002-09-15", 20)
        '2002-10-05'
    """
    return str(TemporalDate.parse(date_str).add_days(days))


def parse_date(date_str: str) -> TemporalDate:
    """
    解析日期字符串
    
    Args:
        date_str: 日期字符串
        
    Returns:
        TemporalDate 对象
    """
    return TemporalDate.parse(date_str)

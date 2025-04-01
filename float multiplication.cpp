#include <iostream>
#include <vector>
#include <cmath>       // Для математических функций (log, exp, logl, expl, fabs, isfinite, isnan, isinf)
#include <random>      // Для генерации случайных чисел (std::mt19937, std::uniform_real_distribution)
#include <algorithm>   // Для алгоритмов, таких как std::shuffle (перемешивание)
#include <iomanip>     // Для манипуляторов вывода (std::fixed, std::setprecision)
#include <limits>      // Для доступа к свойствам числовых типов (std::numeric_limits)
#include <cstdint>     // Для типов целых чисел фиксированного размера (uint64_t)
#include <cstring>     // Для функции memcpy (копирование памяти)
#include <gmp.h>       // Заголовочный файл библиотеки GMP для высокоточной арифметики

// --- Конфигурация ---
// Количество генерируемых чисел для вычисления произведения
const size_t NUM_VALUES = 1000;
// Минимальное значение генерируемых чисел (больше 0 для логарифма)
const double MIN_VALUE = 0.8;
// Максимальное значение генерируемых чисел
const double MAX_VALUE = 1.2;
// Точность (в битах), используемая для вычислений в GMP (для эталонного значения)
const unsigned long GMP_PRECISION_BITS = 256;

union DoubleUnion {
    double d;
    uint64_t i;
};

// --- Прототипы функций ---

// Вычисляет произведение "наивным" способом, используя long double для промежуточных вычислений
double naive_product_ld(const std::vector<double>& data);

// Выполняет суммирование чисел в векторе с использованием алгоритма Кахана для повышения точности. Версия для long double.
long double kahan_sum_ld(const std::vector<long double>& data);

// Вычисляет произведение через сумму логарифмов (log-sum-exp), используя long double для промежуточных вычислений и суммирование Кахана.
double log_sum_exp_product_ld(const std::vector<double>& data);

// Вычисляет эталонное произведение с высокой точностью с помощью библиотеки GMP.
void gmp_product(mpf_t result, const std::vector<double>& data);

// Вычисляет расстояние в ULP (Units in the Last Place) между двумя числами double. Показывает, сколько представимых чисел находится между a и b.
int64_t ulp_distance(double a, double b);

// Печатает результаты вычислений (наивный, log-sum-exp, GMP) и их сравнение по ULP.
void print_results(const std::string& label, // Метка для идентификации (например, "Исходный порядок")
                   double naive_prod,        // Результат наивного произведения (версия LD)
                   double logexp_prod,       // Результат log-sum-exp (версия LD)
                   const mpf_t gmp_ref_prod); // Эталонное значение GMP


int main() {
    std::cout << "Точность double: " << std::numeric_limits<double>::digits << " бит мантиссы ("
              << std::numeric_limits<double>::max_digits10 << " значащих десятичных цифр)" << std::endl;
    std::cout << "Точность long double: " << std::numeric_limits<long double>::digits << " бит мантиссы ("
              << std::numeric_limits<long double>::max_digits10 << " значащих десятичных цифр)" << std::endl;
    bool long_double_is_better = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;
    if (!long_double_is_better) {
        std::cout << "*** Предупреждение: long double не имеет большей точности, чем double на этой платформе! Улучшения точности от его использования не ожидается. ***" << std::endl;
    }
    std::cout << std::endl;

    mpf_set_default_prec(GMP_PRECISION_BITS);

    std::cout << std::fixed
              << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "--- Программа сравнения точности произведения чисел ---" << std::endl;
    std::cout << "Количество чисел: " << NUM_VALUES << std::endl;
    std::cout << "Диапазон чисел: [" << MIN_VALUE << ", " << MAX_VALUE << "]" << std::endl;
    std::cout << "Точность GMP: " << GMP_PRECISION_BITS << " бит" << std::endl << std::endl;

    std::vector<double> numbers(NUM_VALUES);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> dist(MIN_VALUE, MAX_VALUE);
    for (size_t i = 0; i < NUM_VALUES; ++i) {
        numbers[i] = dist(rng);
    }
    std::cout << "Сгенерировано " << numbers.size() << " чисел." << std::endl;

    std::cout << "\n--- Расчет с исходным порядком чисел ---" << std::endl;
    double naive_prod1 = naive_product_ld(numbers);
    double logexp_prod1 = log_sum_exp_product_ld(numbers);

    mpf_t gmp_prod_ref;
    mpf_init(gmp_prod_ref);
    gmp_product(gmp_prod_ref, numbers);

    print_results("Исходный порядок", naive_prod1, logexp_prod1, gmp_prod_ref);

    std::cout << "\n--- Перемешивание данных ---" << std::endl;
    std::vector<double> shuffled_numbers = numbers;
    std::shuffle(shuffled_numbers.begin(), shuffled_numbers.end(), rng);
    std::cout << "Данные перемешаны." << std::endl;

    std::cout << "\n--- Расчет с перемешанным порядком чисел ---" << std::endl;
    double naive_prod2 = naive_product_ld(shuffled_numbers);
    double logexp_prod2 = log_sum_exp_product_ld(shuffled_numbers);

    print_results("Перемешанный порядок", naive_prod2, logexp_prod2, gmp_prod_ref);

    std::cout << "\n--- Проверка инвариантности к перестановке ---" << std::endl;
    int64_t naive_perm_diff = ulp_distance(naive_prod1, naive_prod2);
    if (naive_perm_diff == 0) {
        std::cout << "Наивное произведение (LD): Результат НЕ изменился после перестановки." << std::endl;
    } else {
        std::cout << "Наивное произведение (LD): Результат ИЗМЕНИЛСЯ после перестановки." << std::endl;
        std::cout << "  Разница ULP: " << naive_perm_diff << std::endl;
        std::cout << "  Исходный: " << naive_prod1 << std::endl;
        std::cout << "  Перемеш.: " << naive_prod2 << std::endl;
    }

    int64_t logexp_perm_diff = ulp_distance(logexp_prod1, logexp_prod2);
    if (logexp_perm_diff == 0) {
        std::cout << "Log-Sum-Exp (LD) произведение: Результат НЕ изменился после перестановки." << std::endl;
    } else {
        std::cout << "Log-Sum-Exp (LD) произведение: Результат ИЗМЕНИЛСЯ после перестановки." << std::endl;
        std::cout << "  Разница ULP: " << logexp_perm_diff << std::endl;
        std::cout << "  Исходный: " << logexp_prod1 << std::endl;
        std::cout << "  Перемеш.: " << logexp_prod2 << std::endl;
    }

    mpf_clear(gmp_prod_ref);

    return 0;
}

//Вычисляет произведение элементов вектора data "наивным" способом.

double naive_product_ld(const std::vector<double>& data) {
    long double prod_ld = 1.0L;

    for (double val : data) {
        prod_ld *= static_cast<long double>(val);

        if (!std::isfinite(prod_ld)) {
            break;
        }
    }

    double final_result = static_cast<double>(prod_ld);

    if (!std::isfinite(final_result)) {
        if (std::isinf(prod_ld)) return std::numeric_limits<double>::infinity() * (prod_ld > 0 ? 1.0 : -1.0);
        return std::numeric_limits<double>::quiet_NaN();
    }
    return final_result;
}

//Реализует алгоритм суммирования Кахана для вектора чисел long double.

long double kahan_sum_ld(const std::vector<long double>& data) {
    long double sum = 0.0L;
    long double c = 0.0L;

    for (long double val : data) {
        if (!std::isfinite(val)) {
            continue;
        }
        long double y = val - c;
        long double t = sum + y;

        if (!std::isfinite(t)) {
            return std::numeric_limits<long double>::quiet_NaN();
        }

        c = (t - sum) - y;
        sum = t;
    }

    if (!std::isfinite(sum)) {
    }
    return sum;
}

//Вычисляет произведение элементов через экспоненту от суммы логарифмов.

double log_sum_exp_product_ld(const std::vector<double>& data) {
    std::vector<long double> log_data;
    log_data.reserve(data.size());

    for (double val : data) {
        if (val <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        long double log_val = logl(val);

        if (!std::isfinite(log_val)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        log_data.push_back(log_val);
    }

    long double sum_of_logs = kahan_sum_ld(log_data);

    if (!std::isfinite(sum_of_logs)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    long double result_ld = expl(sum_of_logs);

    if (!std::isfinite(result_ld)) {
        if (std::isinf(result_ld)) return std::numeric_limits<double>::infinity() * (result_ld > 0 ? 1.0 : -1.0);
        if (result_ld == 0.0L) return 0.0;
        return std::numeric_limits<double>::quiet_NaN();
    }

    return static_cast<double>(result_ld);
}

//Вычисляет произведение элементов вектора data с высокой точностью с помощью GMP

void gmp_product(mpf_t result, const std::vector<double>& data) {
    mpf_set_d(result, 1.0);
    mpf_t temp_val;
    mpf_init(temp_val);

    for (double val : data) {
        if (!std::isfinite(val)) {
            continue;
        }
        mpf_set_d(temp_val, val);
        mpf_mul(result, result, temp_val);
    }

    mpf_clear(temp_val);
}

//Вычисляет расстояние в ULP (Units in the Last Place) между двумя числами double.

int64_t ulp_distance(double a, double b) {
    bool a_nan = std::isnan(a);
    bool b_nan = std::isnan(b);
    if (a_nan || b_nan) {
        return std::numeric_limits<int64_t>::max();
    }

    bool a_inf = std::isinf(a);
    bool b_inf = std::isinf(b);
    if (a_inf || b_inf) {
        if (a == b) return 0;
        return std::numeric_limits<int64_t>::max();
    }

    if (a == 0.0 && b == 0.0) {
        return 0;
    }

    DoubleUnion uA, uB;
    std::memcpy(&uA.i, &a, sizeof(double));
    std::memcpy(&uB.i, &b, sizeof(double));

    if ((uA.i >> 63) != (uB.i >> 63)) {
        return std::numeric_limits<int64_t>::max();
    }

    int64_t diff;
    if (uA.i > uB.i) {
        diff = static_cast<int64_t>(uA.i - uB.i);
    } else {
        diff = static_cast<int64_t>(uB.i - uA.i);
    }
    return diff;
}

//Печатает результаты вычислений и сравнение точности.

void print_results(const std::string& label,
                   double naive_prod,
                   double logexp_prod,
                   const mpf_t gmp_ref_prod)
{
    std::cout << "[" << label << "]" << std::endl;

    double gmp_prod_double = mpf_get_d(gmp_ref_prod);

    std::cout << "  Наивное произведение (LD): " << naive_prod << std::endl;
    std::cout << "  Log-Sum-Exp (LD) пр.: " << logexp_prod << std::endl;
    std::cout << "  Эталонное GMP (double): " << gmp_prod_double << std::endl;

    char* gmp_str = nullptr;
    mp_exp_t exp_val;
    gmp_asprintf(&gmp_str, "%.*Ff", (int)mpf_get_default_prec() / 3 + 10, gmp_ref_prod);
    std::cout << "  Эталонное GMP (полн.): ";
    if (gmp_str) {
        std::cout << gmp_str << std::endl;
        free(gmp_str);
    } else {
        std::cout << "[Ошибка вывода GMP]" << std::endl;
    }

    std::cout << "--- Сравнение точности (" << label << ") ---" << std::endl;
    int64_t ulp_naive = ulp_distance(naive_prod, gmp_prod_double);
    int64_t ulp_logexp = ulp_distance(logexp_prod, gmp_prod_double);

    std::cout << "  Разница ULP (Наивный LD vs GMP)  : " << ulp_naive << std::endl;
    std::cout << "  Разница ULP (LogSumExp LD vs GMP): " << ulp_logexp << std::endl;
}

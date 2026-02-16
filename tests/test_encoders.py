import tempfile
import unittest

import torch
from loguru import logger
from tqdm.auto import tqdm

from justatom.modeling.prime import ILanguageModel
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import RuntimeProcessor
from justatom.processing.tokenizer import ITokenizer
from justatom.running.encoders import EncoderRunner
from justatom.running.mask import IModelRunner


class M1LMRunnerTest(unittest.TestCase):
    TEXTS = [
        """
        I'm tired of being what you want me to be
Feeling so faithless lost under the surface
Don't know what you're expecting of me
Put under the pressure of walking in your shoes
(Caught in the undertow just caught in the undertow)
Every step that I take is another mistake to you
(Caught in the undertow just caught in the undertow)

I've become so numb I can't feel you there
I've become so tired so much more aware
I'm becoming this all I want to do
Is be more like me and be less like you

Can't you see that you're smothering me
Holding too tightly afraid to lose control
Cause everything that you thought I would be
Has fallen apart right in front of you
(Caught in the undertow just caught in the undertow)
Every step that I take is another mistake to you
(Caught in the undertow just caught in the undertow)
And every second I waste
Is more than I can take

I've become so numb I can't feel you there
I've become so tired so much more aware
I'm becoming this all I want to do
Is be more like me and be less like you

And I know
I may end up failing too
But I know
You were just like me with someone disappointed in you

I've become so numb I can't feel you there
I've become so tired so much more aware
I'm becoming this all I want to do
Is be more like me and be less like you

I've become so numb I can't feel you there
(I'm tired of being what you want me to be)
I've become so numb I can't feel you there
(I'm tired of being what you want me to be)
        """,
        """
What I've Done
текст песни Linkin Park
Видео песни
In this farewell
There's no blood
There's no alibi
Cause I've drawn regret
From the truth
Of a thousand lies

So let mercy come
And wash away

What I've done
I'll face myself
To cross out what I've become
Erase myself
And let go of what I've done

Put to rest
What you thought of me
Well I've clean this slate
With the hands
Of uncertainty

So let mercy come
And wash away

What I've done
I'll face myself
To cross out what I've become
Erase myself
And let go of what I've done

For what I've done

I start again
And whatever pain may come
Today this ends
I'm forgiving what I've done

I'll face myself
To cross out what I've become
Erase myself
And let go of what I've done

What I've done

Forgiving what I've done
""",
        """
Я вспомню свои 20: темы, тусовки, тёрки
Кутузу разрезает переодетая в "Мку" пятёрка
Меня девчонка ждет дома, я с центра на тренировку
Утром в храм с пацанами, в моей душе всё по полкам
Я бил и пропускал, извинялся и прощал
Был предан, но видит Бог, я никогда не предавал
Мог разбиться, но пролетал, сдаться, но газовал
Хотел влюбиться тыщу раз, но влюблялся когда не ждал
Я менялся и вырастал, люди шли, я их отпускал
Обещав себе ночами: "Мой шлягер споет квартал!"
Ставил и поднимал, потом ставил и прогорал
А время шло и я видел как Бог давал
Дорогу идущему на закат
Покой тем, кто не встал с новым рассветом
Демоны сбивают нас с координат
Но мы возвращаемся на сторону света
""",
        """
Мишленовские шины, спортивные машины, я
Спортики в машине, мы обгоняем машины
Запах анаши и телефоны в авиарежиме
Твой вес — мой рабочий вес, мы резко стали большими
Я щас взгреваю всех, кто в детстве в меня вложили
У на— У нас щас успех, но когда-то мы так не жили
Базару нет, ты смотришь прямо, но я смотрю шире
Они пиздели чёто за огонь, но мы их потушили (В салоне +21)
""",
        """
        Тут патрон в патроннике, ежели что
Головы рубили, зарабатывал вес
Тут каждый, кто бил, добивая, нашёл
Личную выгоду и свой интерес
Накал пламенел, озарял города
Неминуемая яма лыбу давила ехидно
Тут человек пропадал навсегда
В самом себе себя же не было видно
        """,
        """
        Тут душу греет водный, не молодеет бодрый
Еб- тут будет недовольный у любого и не без контры
Тут судят люд гордый, я расковыряю критику твою минором
Ибо не- тут ловить вовсе
Чистилище сего губительного называй
Не приведи, Санта, мне менять эту игру на рай
Я с друидами дружил и знаю чё по чём
И в подворотнях этих грязных убивался горячо
Я без малого района Мартин Лютер Кинг
Не миновала меня палёная заруба некоренных
Жителей миокарда, вам меня не надо
Искренне вам желаю меня не понять и потерять
Правда, исповедей тут, как плюса, нелюдимым не видать
Тёмники тут лавандосом разменяли благодать
Там ревели горы, затмевая чудеса (а)
Ревели наши голоса
        """,
    ]

    def setUp(self):

        model_name_or_path = "intfloat/multilingual-e5-base"

        tokenizer = ITokenizer.from_pretrained(model_name_or_path)
        processor = RuntimeProcessor(tokenizer)
        lm_model = ILanguageModel.load(model_name_or_path)

        self.runner = EncoderRunner(
            model=lm_model,
            prediction_heads=[],
            processor=processor,
        )
        self.runner.eval()

    def test_inference(self):
        dataset, tensor_names, problematic_ids = (
            self.runner.processor.dataset_from_dicts(  # pyright: ignore[reportOptionalMemberAccess]
                [{"content": content} for content in self.TEXTS]
            )
        )
        loader = NamedDataLoader(
            dataset=dataset, tensor_names=tensor_names, batch_size=2
        )
        vectors = []
        for batch in tqdm(loader):  # Each batch comes with bs=2 samples
            with torch.no_grad():
                vecs = self.runner(batch)[0]
                vectors.extend(vecs)
        for i in range(1, len(vectors), 2):
            dot_product_positive = vectors[i] @ vectors[i - 1]
            dot_product_negative = vectors[i] @ vectors[(i + 1) % len(vectors)]
            logger.info(
                f"i={i} | dot_product_positive={dot_product_positive} | dot_product_negative={dot_product_negative}"
            )
            self.assertGreater(dot_product_positive, dot_product_negative)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash

# Скрипт по загрузке публичных файлов с Облака mail.ru. Поддерживается докачка.
# v1.0.5 от 2022-05-30
#
# ЕСЛИ СКРИПТ НЕ РАБОТАЕТ
#
# 1. Убедитесь, что файл доступен публично. Возможна загрузка только публичных файлов.
# 2. Mail.ru время от времени меняет внутрянку, не очень сильно, но требуется адаптация скрипта.
#    Если скрипт не работает, просьба сделать работающий форк и скинуть ссылку в комментарии.
#    Спасибо.
#
# КАК ПОЛЬЗОВАТЬСЯ
#
# 1. Скачать скрипт.
# 2. Сделать его исполняемым:
# 	chmod +x dl-cloud-mail-ru.sh
# 3. Получить ссылку на файл для скачивания (см. чуть ниже, как это сделать).
# 3. Скачать файл из облака:
# 	./dl-cloud-mail-ru.sh ссылка_на_файл локальный_путь
#    Например (пример рабочий, файл существует, можно потестировать скрипт):
# 	./dl-cloud-mail-ru.sh https://cloud.mail.ru/public/Y5C8/KRwhz4JHW/linux-5.12.7.tar.xz linux-kernel.tar.xz
#
# КАК ПОЛУЧИТЬ ССЫЛКУ НА СКАЧИВАНИЕ ФАЙЛА В ПАПКЕ
#
# Если у вас есть ссылка на папку, и нужно скачать файл оттуда:
#
# 1. Два раза кликнуть на файл в папке. Появится всплывающее окно с кнопкой "Скачать".
# 2. В это время в адресной строке браузера будет отображаться ссылка на файл. Это и есть ссылка для скрипта.

# ENGLISH
#
# IMPORTANT: mail.ru sometimes changes internals, not too much, but script must be changed.
#
# If this script does not work:
#   - see forks, may be there is a fix already,
#   - if not, please post patch in comments or create a working fork of this gist.
# Thank you!

# ИСТОРИЯ И БЛАГОДАРНОСТИ
#
# 2022-05-30 отражены изменения со стороны mail.ru, спасибо https://gist.github.com/grumbik
# 2021-05-27 дополнена документация
# 2021-05-26 изменения со стороны mail.ru, плюс теперь определяем url к файлу проще без обращения к api,
#            спасибо kerastinell https://gist.github.com/kerastinell
# 2018-06-18 mail.ru изменил формат страницы
# 2017-09-22 исходная идея: https://novall.net/itnews/bash-skript-dlya-skachivaniya-fajlov-s-mail-ru-cherez-konsol-linux.html

URL="$1"
DST_FILE="$2"

[ -z "$DST_FILE" ] && {
    echo "Syntax: `basename $0` <url> <dst_file>" >&2
    echo "Example: `basename $0` https://cloud.mail.ru/public/BeAr/3s8QfYgLj /path/to/my/file.rar" >&2
    echo "Test: `basename $0` https://cloud.mail.ru/public/Y5C8/KRwhz4JHW/linux-5.12.7.tar.xz linux-5.12.7.tar.xz" >&2
    exit 1
}

function getPageInformation() {
	local pageUrl="$1"

	wget --quiet -O - "$pageUrl" > page.html
	wget --quiet -O - "$pageUrl" | sed -n "/window.cloudSettings/,/};<\/script>/p"
}

function ensureFileExists() {
	local pageInformation="$1"

	echo "$pageInformation" |  grep -q '"not_exists"' && {
		echo "Error: file does not exist" >&2
		exit 1
	}
}

function extractDownloadUrl() {
	local pageUrl="$1" pageInformation="$2" storageUrl filePath

	storageUrl=$(echo "$pageInformation" | sed -r 's|</?script[^>]*>|\n|g' | sed -n 's|.*\(window.cloudSettings=.*}\)|\1|p' | sed -n 's/.*\(weblink_get[^}]*}\).*/\1/p' | sed -n 's|.*\(https://[^"]*\)".*|\1|p')
  filePath=$(echo "$pageUrl" | awk -F '/public/' '{print $2}')

	[ -z "$storageUrl" ] || [ -z "$filePath" ] && {
		echo "Error: failed to extract storage's url or file path" >&2
		exit 1
	}

	echo "$storageUrl/$filePath"
}

pageInformation=$(getPageInformation "$URL")
ensureFileExists "$pageInformation"
downloadUrl=$(extractDownloadUrl "$URL" "$pageInformation")

wget --continue --no-check-certificate --referer="$URL" "$downloadUrl" -O "$DST_FILE"

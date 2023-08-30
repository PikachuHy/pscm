//
// Created by PikachuHy on 2023/7/22.
//
#include <SDL.h>
#include <SDL_render.h>
#include <SDL_ttf.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
namespace fs = std::filesystem;
// import std;
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
namespace krabby {

const static int KEY_HEIGHT = 80;
const char *KEY_TAB = "<tab>";
const char *KEY_BACKSPACE = "<backspace>";
const char *KEY_CAPS_LOCK = "<Caps Lock>";
const char *KEY_ENTER = "<enter>";
const char *KEY_SHIFT = "<shift>";
const char *KEY_RIGHT_SHIFT = "<right-shift>";
const char *KEY_FN = "<fn>";
const char *KEY_CTRL = "<ctrl>";
const char *KEY_OPTION = "<option>";
const char *KEY_COMMAND = "<command>";
const char *KEY_RIGHT_COMMAND = "<right-command>";
const char *KEY_RIGHT_OPTION = "<right-option>";
const char *DEFAULT_KEYBOARD_FONT = "American Typewriter";

class Keyboard {
public:
  struct KeyButton {
    int x;
    int y;
    int w;
    bool highlight;
  };

  Keyboard(SDL_Renderer *renderer, TTF_Font *ch_font, TTF_Font *fn_font)
      : renderer_(renderer)
      , ch_font_(ch_font)
      , fn_font_(fn_font) {
  }

  void init_keyboard() {
    auto f = [this](int x, int y, int w, std::string& line, auto cb) {
      for (auto ch : line) {
        std::string s;
        s += ch;
        btn_map_[s] = KeyButton{ .x = x, .y = y, .w = w, .highlight = false };
        x += KEY_HEIGHT + 10;
      }
      cb(x, y, w);
    };
    // init ` 1 2 ... 9 0 - + delete
    std::string line = "`1234567890-+";
    int x = 0;
    int y = 0;
    int w = KEY_HEIGHT;
    f(x, y, w, line, [this](int x, int y, int w) {
      btn_map_[KEY_BACKSPACE] = KeyButton{
        .x = x,
        .y = y,
        .w = int(w * 1.5),
        .highlight = false,
      };
    });

    line = "qwertyuiop[]\\";
    keyboard_width = line.size() * (KEY_HEIGHT + 10) + 1.5 * KEY_HEIGHT;
    y += KEY_HEIGHT + 10;
    btn_map_[KEY_TAB] = KeyButton{ .x = x, .y = y, .w = int(w * 1.5), .highlight = false };
    f(x + KEY_HEIGHT * 1.5 + 10, y, w, line, [](int x, int y, int w) {
      // do nothing
    });
    line = "asdfghjkl;'";
    y += KEY_HEIGHT + 10;
    btn_map_[KEY_CAPS_LOCK] = KeyButton{ .x = x, .y = y, .w = w * 2 - 10, .highlight = false };
    f(x + KEY_HEIGHT * 2, y, w, line, [this](int x, int y, int w) {
      btn_map_[KEY_ENTER] = KeyButton{
        .x = x,
        .y = y,
        .w = keyboard_width - x,
        .highlight = false,
      };
    });
    line = "zxcvbnm,./";
    y += KEY_HEIGHT + 10;
    btn_map_[KEY_SHIFT] = KeyButton{ .x = x, .y = y, .w = int(w * 2.7), .highlight = false };
    f(x + KEY_HEIGHT * 2.7 + 10, y, w, line, [this](int x, int y, int w) {
      btn_map_[KEY_RIGHT_SHIFT] = KeyButton{
        .x = x,
        .y = y,
        .w = keyboard_width - x,
        .highlight = false,
      };
    });
    y += KEY_HEIGHT + 10;
    std::vector<std::string> l;
    l.emplace_back(KEY_FN);
    l.emplace_back(KEY_CTRL);
    l.emplace_back(KEY_OPTION);
    for (const auto& s : l) {
      btn_map_[s] = KeyButton{ .x = x, .y = y, .w = w, .highlight = false };
      x += KEY_HEIGHT + 10;
    }
    int cmd_width = KEY_HEIGHT * 1.7 - 10;
    btn_map_[KEY_COMMAND] = KeyButton{ .x = x, .y = y, .w = cmd_width, .highlight = false };
    x += cmd_width + 10;
    int space_width = 5 * w + 4 * 10;
    btn_map_[" "] = KeyButton{ .x = x, .y = y, .w = space_width, .highlight = false };
    x += space_width + 10;
    btn_map_[KEY_RIGHT_COMMAND] = KeyButton{ .x = x, .y = y, .w = cmd_width, .highlight = false };
    x += cmd_width + 10;
    btn_map_[KEY_RIGHT_OPTION] = KeyButton{ .x = x, .y = y, .w = w, .highlight = false };
    keyboard_height = y + KEY_HEIGHT + 10;
  }

  void map_to(SDL_Rect& rect) {
    rect.x += 300;
    rect.y += 600;
  }

  void draw() {
    for (const auto& [ch, key] : btn_map_) {
      SDL_Rect rect = { .x = key.x, .y = key.y, .w = key.w, .h = KEY_HEIGHT };
      map_to(rect);
      if (key.highlight) {
        // #008B45
        SDL_SetRenderDrawColor(renderer_, 0x00, 0x8B, 0x45, 0xFF);
        SDL_RenderFillRect(renderer_, &rect);
      }
      else {
        SDL_SetRenderDrawColor(renderer_, 0x00, 0x00, 0x00, 0xFF);
        SDL_RenderDrawRect(renderer_, &rect);
      }
      TTF_Font *font;
      std::string original_text;
      std::string ch_string = "-+[]\\;',./";
      if ((ch.size() == 1 && isalnum(ch.front())) || ch == " " || ch_string.find(ch[0]) != std::string::npos) {
        font = ch_font_;
        original_text = ch;
      }
      else {
        font = fn_font_;

        if (ch == KEY_TAB) {
          original_text = "Tab";
        }
        else if (ch == KEY_BACKSPACE) {
          original_text = "Backspace";
        }
        else if (ch == KEY_CAPS_LOCK) {
          original_text = "Caps Lock";
        }
        else if (ch == KEY_ENTER) {
          original_text = "Enter";
        }
        else if (ch == KEY_SHIFT || ch == KEY_RIGHT_SHIFT) {
          original_text = "Shift";
        }
        else if (ch == KEY_FN) {
          original_text = "fn";
        }
        else if (ch == KEY_CTRL) {
          original_text = "control";
        }
        else if (ch == KEY_OPTION || ch == KEY_RIGHT_OPTION) {
          original_text = "option";
        }
        else if (ch == KEY_COMMAND || ch == KEY_RIGHT_COMMAND) {
          original_text = "command";
        }
        else {
          original_text = ch;
        }
      }
      SDL_Color key_color = { 0, 0, 0 };
      SDL_Surface *text = TTF_RenderUTF8_Blended(font, original_text.c_str(), key_color);

      SDL_Rect textrect = { key.x + (key.w - text->w) / 2, key.y + (KEY_HEIGHT - text->h) / 2, text->w, text->h };
      map_to(textrect);
      SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer_, text);
      SDL_RenderCopy(renderer_, texture, nullptr, &textrect);
      SDL_DestroyTexture(texture);
      SDL_FreeSurface(text);
    }
  }

  void highlight(char ch) {

    for (auto& [k, v] : btn_map_) {
      v.highlight = false;
    }
    if (std::isupper(ch)) {
      ch = std::tolower(ch);
      std::string left = "qwertasdfgzxcvb";
      if (left.find(ch) == std::string::npos) {
        btn_map_[KEY_SHIFT].highlight = true;
      }
      else {
        btn_map_[KEY_RIGHT_SHIFT].highlight = true;
      }
    }
    else {
      std::string left = R"(!@#$%^)";
      std::string left_map = R"(123456)";
      std::string right = R"(&*()_+{}|:"<>?)";
      std::string right_map = R"(7890-=[]\;',./)";
      if (left.find(ch) != std::string::npos) {
        btn_map_[KEY_RIGHT_SHIFT].highlight = true;
        ch = left_map[left.find(ch)];
      }
      else if (right.find(ch) != std::string::npos) {
        btn_map_[KEY_SHIFT].highlight = true;
        ch = right_map[right.find(ch)];
      }
    }
    std::string s;
    s += ch;
    btn_map_[s].highlight = true;
  }

private:
  SDL_Renderer *renderer_;
  TTF_Font *ch_font_;
  TTF_Font *fn_font_;

  std::unordered_map<std::string, KeyButton> btn_map_;
  int keyboard_width;
  int keyboard_height;
};

class TypeWidget {
public:
  TypeWidget(SDL_Renderer *renderer, TTF_Font *font)
      : renderer_(renderer)
      , font_(font) {

    article_ = read_article("asset/article/Speech/I Have a Dream.txt");
    article_ = clean_article(article_);
  }

  std::string read_article(const std::string& path) {
    std::ifstream fin;
    fin.open(path);
    if (!fin.is_open()) {
      std::cout << "read error: " << path << std::endl;
      std::exit(1);
    }
    std::string buffer;
    auto sz = fs::file_size(path);
    buffer.resize(sz);
    fin.read(buffer.data(), sz);
    return buffer;
  }

  std::string clean_article(const std::string& buffer) {
    std::string ret;
    ret.reserve(buffer.size());
    std::string sp_ch_string = "\t\n\r";
    bool last_is_space = false;
    for (auto ch : buffer) {
      if (sp_ch_string.find(ch) == std::string::npos) {
        ret.push_back(ch);
      }
      else {
        if (!last_is_space) {
          ret.push_back(' ');
        }
      }
      last_is_space = ret.back() == ' ';
    }
    return ret;
  }

  bool type_ch(char ch) {
    auto idx = typed_.size();
    typed_.push_back(ch);
    if (idx >= article_.size()) {
      return false;
    }
    return ch == article_.at(idx);
  }

  void draw_string(const std::string& s, const SDL_Color color, int x, int y) {
    SDL_Surface *text = TTF_RenderUTF8_Blended(font_, s.c_str(), color);
    SDL_Rect rect = { x, y, text->w, text->h };
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer_, text);
    SDL_RenderCopy(renderer_, texture, nullptr, &rect);
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(text);
  }

  void delete_last_char() {
    if (typed_.empty()) {
      return;
    }
    typed_.pop_back();
  }

  [[nodiscard]] std::optional<char> current_ch() const {
    if (article_.empty()) {
      return std::nullopt;
    }
    if (typed_.size() >= article_.size()) {
      return std::nullopt;
    }
    return article_.at(typed_.size());
  }

  void draw() {
    int x = 10;
    int y = 10;
    int h = 0;
    for (int i = 0; i < article_.size(); ++i) {
      SDL_Color key_color = { 0, 0, 0 };
      auto cur_ch = std::string(article_.substr(i, 1));
      SDL_Surface *text = TTF_RenderUTF8_Blended(font_, cur_ch.c_str(), key_color);
      auto cur_w = text->w;
      auto cur_h = text->h;
      SDL_Rect textrect = { x, y, text->w, text->h };
      SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer_, text);
      SDL_RenderCopy(renderer_, texture, nullptr, &textrect);
      SDL_DestroyTexture(texture);
      SDL_FreeSurface(text);
      h = std::max(h, cur_h);
      if (typed_.size() > i) {
        if (typed_[i] == article_[i]) {
          key_color = { 0x5B, 0x5B, 0x5B };
        }
        else {
          key_color = { 0xFF, 0, 0 };
        }
        draw_string(typed_.substr(i, 1).c_str(), key_color, x, y + cur_h + 2);
      }
      if (i == typed_.size()) {
        if (show_cursor_) {
          SDL_SetRenderDrawColor(renderer_, 0x00, 0x00, 0x00, 0xFF);
          SDL_Rect rect{ x, y + h + 2, 1, h };
          SDL_RenderDrawRect(renderer_, &rect);
          SDL_RenderDrawLine(renderer_, x - 2, y + h + 2, x + 2, y + h + 2);
          SDL_RenderDrawLine(renderer_, x - 2, y + h + 2 + h, x + 2, y + h + 2 + h);
        }
      }
      x += cur_w;
      if (x >= 1900) {
        x = 10;
        y += h * 2 + 4;
      }
      if (y > 500) {
        break;
      }
    }
  }

  void trigger_cursor() {
    show_cursor_ = !show_cursor_;
  }

private:
  SDL_Renderer *renderer_;
  TTF_Font *font_;
  std::string article_;
  std::string typed_;
  bool show_cursor_ = true;
};
} // namespace krabby

SDL_Window *window;
SDL_Renderer *renderer;
SDL_Rect rect = { .x = 0, .y = 0, .w = 1920, .h = 1080 };
uint32_t ticksForNextKeyDown = 10;
krabby::Keyboard *keyboard;
krabby::TypeWidget *type_widget;

typedef struct AudioBuffer {
  uint32_t len = 0;
  int pullLen = 0;
  uint8_t *data = nullptr;
} AudioBuffer;

AudioBuffer audio_type;
AudioBuffer audio_error;
std::optional<AudioBuffer> cur_audio;

void redraw() {
  SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
  SDL_RenderClear(renderer);
  type_widget->draw();
  auto cur_ch = type_widget->current_ch();
  if (cur_ch.has_value()) {
    keyboard->highlight(cur_ch.value());
  }
  keyboard->draw();
  SDL_RenderPresent(renderer);
}

bool handle_events() {

  SDL_Event event;
  SDL_PollEvent(&event);
  std::optional<bool> has_typed;
  if (event.type == SDL_QUIT) {
    return false;
  }
  if (event.type == SDL_USEREVENT) {
    type_widget->trigger_cursor();
    redraw();
  }
  if (event.type == SDL_KEYDOWN) {
    uint32_t tickNow = SDL_GetTicks();
    if (SDL_TICKS_PASSED(tickNow, ticksForNextKeyDown)) {
      ticksForNextKeyDown = tickNow + 10;
      char keynum = char(event.key.keysym.sym);
      std::string valid_key = " !\"#%$&'()*+,./0123456789:;<=>?@[\\]^_`";
      std::string valid_num_shift = ")!@#$%^&*(";
      if (keynum >= 'a' && keynum <= 'z') {
        if (event.key.keysym.mod & KMOD_SHIFT) {
          has_typed = type_widget->type_ch(keynum - 'a' + 'A');
        }
        else {
          has_typed = type_widget->type_ch(keynum);
        }
      }
      else if (keynum >= '0' && keynum <= '9') {
        if (event.key.keysym.mod & KMOD_SHIFT) {
          has_typed = type_widget->type_ch(valid_num_shift[keynum - '0']);
        }
        else {
          has_typed = type_widget->type_ch(keynum);
        }
      }
      else if (valid_key.find(keynum) != std::string::npos) {
        has_typed = type_widget->type_ch(keynum);
      }
      else if (event.key.keysym.mod & KMOD_SHIFT) {
        // FIXME
        std::map<char, char> key_map = {
          { ';', ':'},
          {'\'', '"'},
          { '[', '{'},
          { ']', '}'},
          { '-', '_'},
          { '=', '+'},
          { ',', '<'},
          { '.', '>'},
          { '/', '?'}
        };
        auto it = key_map.find(keynum);
        if (it != key_map.end()) {
          has_typed = type_widget->type_ch(it->second);
        }
        else {
          std::cout << "not found" << keynum << std::endl;
        }
      }
      else {
        switch (event.key.keysym.sym) {
        case SDLK_BACKSPACE:
          type_widget->delete_last_char();
          break;
        default: {
        }
        }
      }
      redraw();
    }
  }
  if (has_typed.has_value()) {
    if (has_typed.value()) {
      cur_audio = audio_type;
    }
    else {
      cur_audio = audio_error;
    }
  }
  return true;
}

Uint32 timer_callback(Uint32 interval, void *) {
  SDL_Event event;
  event.type = SDL_USEREVENT;
  SDL_PushEvent(&event);
  return interval;
}

void run_main_loop() {
#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop(
      []() {
        handle_events();
      },
      0, true);
#else
  while (handle_events()) {
  }
#endif
}

void fill_audio(void *userdata, Uint8 *stream, int len) {
  SDL_memset(stream, 0, len);
  if (!cur_audio.has_value()) {
    return;
  }
  auto buffer = &cur_audio.value();
  if (buffer->len <= 0) {
    cur_audio.reset();
    return;
  }
  buffer->pullLen = buffer->len > len ? len : buffer->len;
  SDL_MixAudio(stream, buffer->data, buffer->pullLen, SDL_MIX_MAXVOLUME);
  buffer->data += buffer->pullLen;
  buffer->len -= buffer->pullLen;
}

int main() {
  SDL_Init(SDL_INIT_VIDEO);
  TTF_Init();
  std::string artical_font_path = "asset/font/Courier-1.ttf";
  std::string keyboard_font_path = "asset/font/AmericanTypewriter-01.subset.ttf";
  std::string audio_type_path = "asset/audio/type.wav";
  std::string audio_error_path = "asset/audio/error.wav";
  TTF_Font *ch_font = TTF_OpenFont(keyboard_font_path.c_str(), 35);
  TTF_Font *fn_font = TTF_OpenFont(keyboard_font_path.c_str(), 20);
  if (!ch_font) {
    std::cout << "font is nullptr" << std::endl;
    return 0;
  }
  TTF_Font *article_font = TTF_OpenFont(artical_font_path.c_str(), 35);
  SDL_CreateWindowAndRenderer(1920, 1080, 0, &window, &renderer);
  keyboard = new krabby::Keyboard(renderer, ch_font, fn_font);
  keyboard->init_keyboard();
  type_widget = new krabby::TypeWidget(renderer, article_font);
  redraw();
  SDL_AudioSpec spec;
  if (!SDL_LoadWAV(audio_type_path.c_str(), &spec, &audio_type.data, &audio_type.len)) {
    std::cout << "load wav failed: " << audio_type_path << SDL_GetError();
    SDL_Quit();
    return 1;
  }

  if (!SDL_LoadWAV(audio_error_path.c_str(), &spec, &audio_error.data, &audio_error.len)) {
    std::cout << "load wav failed: " << audio_error_path << SDL_GetError();
    SDL_Quit();
    return 1;
  }
  spec.userdata = nullptr;
  spec.callback = fill_audio;
  if (SDL_OpenAudio(&spec, nullptr)) {
    std::cout << "open audio failed" << std::endl;
    SDL_FreeWAV(audio_type.data);
    SDL_FreeWAV(audio_error.data);
    SDL_Quit();
    return 1;
  }

  SDL_PauseAudio(0);
  SDL_TimerID timer_cursor = SDL_AddTimer(1000, timer_callback, nullptr);
  run_main_loop();
  SDL_RemoveTimer(timer_cursor);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  return 0;
}
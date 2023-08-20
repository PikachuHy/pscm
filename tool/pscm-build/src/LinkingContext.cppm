export module pscm.build:LinkingContext;
import std;

export namespace pscm::build {
class LinkingContext {
public:
  class Builder {
  public:
    void add_library(const std::string& library) {
      libraries_.insert(library);
    }

    LinkingContext build() {
      LinkingContext ctx{};
      ctx.libraries_ = std::move(libraries_);
      return ctx;
    }

  private:
    std::unordered_set<std::string> libraries_;
  };

public:
  auto libraries() const -> const std::unordered_set<std::string>& {
    return libraries_;
  }

  void merge(const LinkingContext& ctx) {
    libraries_.insert(ctx.libraries().begin(), ctx.libraries().end());
  }

private:
  std::unordered_set<std::string> libraries_;
};
} // namespace pscm::build
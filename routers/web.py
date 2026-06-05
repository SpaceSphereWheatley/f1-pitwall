import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address

from auth import create_access_token, get_current_user_or_redirect, require_admin
from database import (
    change_password,
    create_user,
    delete_user,
    get_all_users,
    get_all_settings,
    set_setting,
    get_user_by_username,
    verify_password,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")
limiter = Limiter(key_func=get_remote_address)


@router.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/dashboard")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
):
    user = await get_user_by_username(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=401,
        )

    token = create_access_token({"sub": user["username"]})
    resp = RedirectResponse(url="/dashboard", status_code=303)
    resp.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax",
        secure=os.getenv("ENVIRONMENT", "development") == "production",
    )
    return resp


@router.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("access_token")
    return resp


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user=Depends(get_current_user_or_redirect)):
    return templates.TemplateResponse(
        "timing.html",
        {"request": request, "user": dict(current_user), "page": "dashboard"},
    )


@router.get("/overview", response_class=HTMLResponse)
async def overview(request: Request, current_user=Depends(get_current_user_or_redirect)):
    return templates.TemplateResponse(
        "overview.html",
        {"request": request, "user": dict(current_user), "page": "overview"},
    )


@router.get("/strategy", response_class=HTMLResponse)
async def strategy(request: Request, current_user=Depends(get_current_user_or_redirect)):
    return templates.TemplateResponse(
        "strategy.html",
        {"request": request, "user": dict(current_user), "page": "strategy"},
    )


@router.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request, current_user=Depends(get_current_user_or_redirect)):
    return templates.TemplateResponse(
        "predictions.html",
        {"request": request, "user": dict(current_user), "page": "predictions"},
    )


@router.get("/racecontrol", response_class=HTMLResponse)
async def racecontrol(request: Request, current_user=Depends(get_current_user_or_redirect)):
    return templates.TemplateResponse(
        "racecontrol.html",
        {"request": request, "user": dict(current_user), "page": "racecontrol"},
    )


@router.get("/schedule", response_class=HTMLResponse)
async def schedule(request: Request, current_user=Depends(get_current_user_or_redirect)):
    return templates.TemplateResponse(
        "schedule.html",
        {"request": request, "user": dict(current_user), "page": "schedule"},
    )


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request, current_user=Depends(require_admin)):
    users = await get_all_users()
    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "user": dict(current_user),
            "users": [dict(u) for u in users],
            "error": None,
            "success": None,
        },
    )


@router.post("/admin/users/create")
async def admin_create_user(
    request: Request,
    current_user=Depends(require_admin),
    username: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
):
    users = await get_all_users()
    try:
        await create_user(username, password, is_admin)
        users = await get_all_users()
        return templates.TemplateResponse(
            "admin/users.html",
            {
                "request": request,
                "user": dict(current_user),
                "users": [dict(u) for u in users],
                "error": None,
                "success": f"User '{username}' created.",
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "admin/users.html",
            {
                "request": request,
                "user": dict(current_user),
                "users": [dict(u) for u in users],
                "error": str(e),
                "success": None,
            },
        )


@router.post("/admin/users/delete")
async def admin_delete_user(
    request: Request,
    current_user=Depends(require_admin),
    username: str = Form(...),
):
    if username == current_user["username"]:
        users = await get_all_users()
        return templates.TemplateResponse(
            "admin/users.html",
            {
                "request": request,
                "user": dict(current_user),
                "users": [dict(u) for u in users],
                "error": "You cannot delete your own account.",
                "success": None,
            },
        )
    await delete_user(username)
    users = await get_all_users()
    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "user": dict(current_user),
            "users": [dict(u) for u in users],
            "error": None,
            "success": f"User '{username}' deleted.",
        },
    )


@router.post("/admin/users/change-password")
async def admin_change_password(
    request: Request,
    current_user=Depends(require_admin),
    username: str = Form(...),
    new_password: str = Form(...),
):
    await change_password(username, new_password)
    users = await get_all_users()
    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "user": dict(current_user),
            "users": [dict(u) for u in users],
            "error": None,
            "success": f"Password updated for '{username}'.",
        },
    )

@router.get("/admin/settings")
async def admin_settings(request: Request, current_user=Depends(require_admin)):
    settings = await get_all_settings()
    return templates.TemplateResponse("admin/settings.html", {
        "request": request,
        "user": current_user,
        "settings": settings,
        "current_year": datetime.now(timezone.utc).year,
    })


@router.post("/admin/settings")
async def admin_settings_save(
    request: Request,
    current_user=Depends(require_admin),
):
    form = await request.form()
    for key, value in form.items():
        await set_setting(key, value.strip())
    settings = await get_all_settings()
    return templates.TemplateResponse("admin/settings.html", {
        "request": request,
        "user": current_user,
        "settings": settings,
        "saved": True,
        "current_year": datetime.now(timezone.utc).year,
    })
